/**
 * LangSmith Integration Page
 * 
 * Provides a comprehensive interface for managing LangSmith integration including:
 * - Connection status monitoring
 * - Project selection and sync control
 * - Sync statistics and history
 * - Webhook configuration
 * - Real-time sync monitoring
 */

'use client';

import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Cloud, 
  Database, 
  Download, 
  Settings, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  Clock, 
  AlertTriangle,
  Play,
  BarChart3,
  ExternalLink,
  Trash2
} from 'lucide-react';

interface ConnectionStatus {
  connected: boolean;
  error?: string;
  project: string;
  last_sync?: string;
  base_url: string;
  webhook_configured: boolean;
}

interface SyncStats {
  total_langsmith_traces: number;
  recent_synced_count: number;
  last_sync?: string;
  project: string;
  connected: boolean;
  error?: string;
}

interface LangSmithProject {
  id: string;
  name: string;
  description?: string;
  created_at?: string;
  example_count: number;
}

interface SyncResult {
  status: string;
  project_name: string;
  total_synced: number;
  last_sync?: string;
  errors: string[];
}

export default function LangSmithPage() {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus | null>(null);
  const [syncStats, setSyncStats] = useState<SyncStats | null>(null);
  const [projects, setProjects] = useState<LangSmithProject[]>([]);
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [loading, setLoading] = useState({
    status: false,
    sync: false,
    projects: false,
    test: false
  });
  const [syncResult, setSyncResult] = useState<SyncResult | null>(null);
  const [syncLimit, setSyncLimit] = useState(100);
  const [forceResync, setForceResync] = useState(false);

  // Fetch connection status and sync stats
  const fetchStatus = async () => {
    setLoading(prev => ({ ...prev, status: true }));
    try {
      const response = await fetch('/api/v1/langsmith/status', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setConnectionStatus(data.connection);
        setSyncStats(data.sync_stats);
      } else {
        console.error('Failed to fetch LangSmith status');
      }
    } catch (error) {
      console.error('Error fetching LangSmith status:', error);
    } finally {
      setLoading(prev => ({ ...prev, status: false }));
    }
  };

  // Fetch available projects
  const fetchProjects = async () => {
    setLoading(prev => ({ ...prev, projects: true }));
    try {
      const response = await fetch('/api/v1/langsmith/projects', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setProjects(data);
        if (data.length > 0 && !selectedProject) {
          setSelectedProject(data[0].name);
        }
      } else {
        console.error('Failed to fetch LangSmith projects');
      }
    } catch (error) {
      console.error('Error fetching LangSmith projects:', error);
    } finally {
      setLoading(prev => ({ ...prev, projects: false }));
    }
  };

  // Perform sync operation
  const performSync = async (background = false) => {
    setLoading(prev => ({ ...prev, sync: true }));
    setSyncResult(null);
    
    try {
      const endpoint = background ? '/api/v1/langsmith/sync/background' : '/api/v1/langsmith/sync';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          project_name: selectedProject || undefined,
          limit: syncLimit,
          force_resync: forceResync
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (background) {
          setSyncResult({
            status: 'background_started',
            project_name: data.project_name,
            total_synced: 0,
            errors: []
          });
          // Refresh status after a delay
          setTimeout(fetchStatus, 3000);
        } else {
          setSyncResult(data);
          // Refresh status after sync
          fetchStatus();
        }
      } else {
        const errorData = await response.json();
        setSyncResult({
          status: 'error',
          project_name: selectedProject || 'unknown',
          total_synced: 0,
          errors: [errorData.detail || 'Sync failed']
        });
      }
    } catch (error) {
      setSyncResult({
        status: 'error',
        project_name: selectedProject || 'unknown',
        total_synced: 0,
        errors: [`Network error: ${error}`]
      });
    } finally {
      setLoading(prev => ({ ...prev, sync: false }));
    }
  };

  // Test connection
  const testConnection = async () => {
    setLoading(prev => ({ ...prev, test: true }));
    try {
      const response = await fetch('/api/v1/langsmith/test-connection', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        // Refresh status to show updated connection info
        fetchStatus();
        alert(data.message);
      } else {
        const errorData = await response.json();
        alert(`Connection test failed: ${errorData.detail}`);
      }
    } catch (error) {
      alert(`Connection test error: ${error}`);
    } finally {
      setLoading(prev => ({ ...prev, test: false }));
    }
  };

  // Clear sync cache
  const clearSyncCache = async () => {
    try {
      const response = await fetch('/api/v1/langsmith/sync-cache', {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      
      if (response.ok) {
        alert('Sync cache cleared successfully');
        fetchStatus();
      } else {
        const errorData = await response.json();
        alert(`Failed to clear cache: ${errorData.detail}`);
      }
    } catch (error) {
      alert(`Error clearing cache: ${error}`);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchProjects();
  }, []);

  // Auto-refresh status every 30 seconds
  useEffect(() => {
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (connected: boolean) => {
    return connected ? (
      <CheckCircle className="w-5 h-5 text-green-500" />
    ) : (
      <XCircle className="w-5 h-5 text-red-500" />
    );
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                <Cloud className="w-8 h-8 mr-3 text-blue-600" />
                LangSmith Integration
              </h1>
              <p className="text-gray-600 mt-2">
                Manage LangSmith synchronization and monitor trace integration
              </p>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={testConnection}
                disabled={loading.test}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {loading.test ? (
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Activity className="w-4 h-4 mr-2" />
                )}
                Test Connection
              </button>
              <button
                onClick={fetchStatus}
                disabled={loading.status}
                className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50"
              >
                {loading.status ? (
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4 mr-2" />
                )}
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Connection Status Card */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Connection Status</h2>
            {connectionStatus && getStatusIcon(connectionStatus.connected)}
          </div>
          
          {connectionStatus ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Status</div>
                <div className={`text-lg font-semibold ${
                  connectionStatus.connected ? 'text-green-600' : 'text-red-600'
                }`}>
                  {connectionStatus.connected ? 'Connected' : 'Disconnected'}
                </div>
                {connectionStatus.error && (
                  <div className="text-xs text-red-500 mt-1">{connectionStatus.error}</div>
                )}
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Project</div>
                <div className="text-lg font-semibold text-gray-900">
                  {connectionStatus.project || 'Not configured'}
                </div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Last Sync</div>
                <div className="text-lg font-semibold text-gray-900">
                  {formatDate(connectionStatus.last_sync)}
                </div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Webhooks</div>
                <div className={`text-lg font-semibold ${
                  connectionStatus.webhook_configured ? 'text-green-600' : 'text-yellow-600'
                }`}>
                  {connectionStatus.webhook_configured ? 'Configured' : 'Not configured'}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <RefreshCw className="w-8 h-8 mx-auto text-gray-400 animate-spin" />
              <p className="text-gray-500 mt-2">Loading connection status...</p>
            </div>
          )}
        </div>

        {/* Sync Statistics Card */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Sync Statistics</h2>
            <BarChart3 className="w-5 h-5 text-blue-600" />
          </div>
          
          {syncStats ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="text-sm text-blue-600">Total LangSmith Traces</div>
                <div className="text-2xl font-bold text-blue-900">
                  {syncStats.total_langsmith_traces.toLocaleString()}
                </div>
              </div>
              
              <div className="bg-green-50 p-4 rounded-lg">
                <div className="text-sm text-green-600">Recent Synced (24h)</div>
                <div className="text-2xl font-bold text-green-900">
                  {syncStats.recent_synced_count.toLocaleString()}
                </div>
              </div>
              
              <div className="bg-purple-50 p-4 rounded-lg">
                <div className="text-sm text-purple-600">Sync Health</div>
                <div className={`text-2xl font-bold ${
                  syncStats.connected ? 'text-green-900' : 'text-red-900'
                }`}>
                  {syncStats.connected ? 'Healthy' : 'Issues'}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-gray-500">Loading sync statistics...</p>
            </div>
          )}
        </div>

        {/* Sync Control Panel */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Sync Control</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Project Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Project Selection
              </label>
              <select
                value={selectedProject}
                onChange={(e) => setSelectedProject(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loading.projects}
              >
                {projects.map((project) => (
                  <option key={project.id} value={project.name}>
                    {project.name} ({project.example_count} examples)
                  </option>
                ))}
              </select>
              {loading.projects && (
                <p className="text-sm text-gray-500 mt-1">Loading projects...</p>
              )}
            </div>

            {/* Sync Options */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Sync Options
              </label>
              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-gray-600">Sync Limit</label>
                  <input
                    type="number"
                    value={syncLimit}
                    onChange={(e) => setSyncLimit(parseInt(e.target.value) || 100)}
                    min="1"
                    max="1000"
                    className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="forceResync"
                    checked={forceResync}
                    onChange={(e) => setForceResync(e.target.checked)}
                    className="mr-2"
                  />
                  <label htmlFor="forceResync" className="text-xs text-gray-600">
                    Force re-sync existing traces
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Sync Buttons */}
          <div className="mt-6 flex space-x-4">
            <button
              onClick={() => performSync(false)}
              disabled={loading.sync || !connectionStatus?.connected}
              className="flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {loading.sync ? (
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Download className="w-4 h-4 mr-2" />
              )}
              Sync Now
            </button>
            
            <button
              onClick={() => performSync(true)}
              disabled={loading.sync || !connectionStatus?.connected}
              className="flex items-center px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              <Play className="w-4 h-4 mr-2" />
              Background Sync
            </button>
            
            <button
              onClick={clearSyncCache}
              className="flex items-center px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Clear Cache
            </button>
          </div>

          {/* Sync Result */}
          {syncResult && (
            <div className={`mt-6 p-4 rounded-lg ${
              syncResult.status === 'success' ? 'bg-green-50 border-green-200' :
              syncResult.status === 'background_started' ? 'bg-blue-50 border-blue-200' :
              'bg-red-50 border-red-200'
            } border`}>
              <div className="flex items-center mb-2">
                {syncResult.status === 'success' ? (
                  <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                ) : syncResult.status === 'background_started' ? (
                  <Clock className="w-5 h-5 text-blue-500 mr-2" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
                )}
                <span className="font-medium">
                  {syncResult.status === 'success' ? 'Sync Completed' :
                   syncResult.status === 'background_started' ? 'Background Sync Started' :
                   'Sync Failed'}
                </span>
              </div>
              
              <div className="text-sm text-gray-700">
                <p>Project: {syncResult.project_name}</p>
                {syncResult.status !== 'background_started' && (
                  <p>Synced: {syncResult.total_synced} traces</p>
                )}
                {syncResult.last_sync && (
                  <p>Completed: {formatDate(syncResult.last_sync)}</p>
                )}
              </div>
              
              {syncResult.errors.length > 0 && (
                <div className="mt-2">
                  <p className="text-sm font-medium text-red-700">Errors:</p>
                  <ul className="text-sm text-red-600 ml-4">
                    {syncResult.errors.map((error, index) => (
                      <li key={index}>• {error}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <a
              href={connectionStatus?.base_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center p-4 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
            >
              <ExternalLink className="w-5 h-5 text-blue-600 mr-3" />
              <div>
                <div className="font-medium text-blue-900">Open LangSmith</div>
                <div className="text-sm text-blue-600">View in dashboard</div>
              </div>
            </a>
            
            <button
              onClick={() => window.open('/docs#langsmith', '_blank')}
              className="flex items-center p-4 bg-green-50 rounded-lg hover:bg-green-100 transition-colors"
            >
              <Settings className="w-5 h-5 text-green-600 mr-3" />
              <div>
                <div className="font-medium text-green-900">API Documentation</div>
                <div className="text-sm text-green-600">Integration guide</div>
              </div>
            </button>
            
            <button
              onClick={() => window.location.href = '/api/v1/traces'}
              className="flex items-center p-4 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors"
            >
              <Database className="w-5 h-5 text-purple-600 mr-3" />
              <div>
                <div className="font-medium text-purple-900">View Traces</div>
                <div className="text-sm text-purple-600">Browse synced data</div>
              </div>
            </button>
            
            <button
              onClick={() => window.location.href = '/analytics'}
              className="flex items-center p-4 bg-orange-50 rounded-lg hover:bg-orange-100 transition-colors"
            >
              <BarChart3 className="w-5 h-5 text-orange-600 mr-3" />
              <div>
                <div className="font-medium text-orange-900">Analytics</div>
                <div className="text-sm text-orange-600">Performance insights</div>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
} 