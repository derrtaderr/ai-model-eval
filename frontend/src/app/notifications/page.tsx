"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Bell, 
  Settings, 
  Send, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Users,
  MessageSquare,
  Activity,
  Zap,
  TestTube2,
  BarChart3,
  Shield
} from 'lucide-react';

interface NotificationPreferences {
  team_id: string;
  enabled: boolean;
  webhook_configured: boolean;
  default_channel?: string;
  notification_types: string[];
  priority_filter: string;
  quiet_hours_start?: number;
  quiet_hours_end?: number;
  rate_limit_per_hour: number;
  mention_users: string[];
  last_updated: string;
}

interface NotificationType {
  type: string;
  name: string;
  description: string;
  priority: string;
  icon: string;
}

interface SlackStatus {
  integration_enabled: boolean;
  webhook_status: string;
  webhook_configured: boolean;
  notification_types_count: number;
  total_notification_types: number;
  rate_limit_per_hour: number;
  quiet_hours_configured: boolean;
  mentions_configured: number;
  last_check: string;
  team_id: string;
}

const PRIORITY_LEVELS = [
  { value: 'low', label: 'Low', color: 'bg-gray-100' },
  { value: 'medium', label: 'Medium', color: 'bg-blue-100' },
  { value: 'high', label: 'High', color: 'bg-orange-100' },
  { value: 'critical', label: 'Critical', color: 'bg-red-100' }
];

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'connected':
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case 'error':
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    case 'not_configured':
      return <Clock className="h-4 w-4 text-gray-400" />;
    default:
      return <Clock className="h-4 w-4 text-gray-400" />;
  }
};

const getNotificationIcon = (iconString: string) => {
  switch (iconString) {
    case '🚨': return <Shield className="h-4 w-4" />;
    case '✅': return <CheckCircle className="h-4 w-4" />;
    case '🧪': return <TestTube2 className="h-4 w-4" />;
    case '📊': return <BarChart3 className="h-4 w-4" />;
    case '⚠️': return <AlertCircle className="h-4 w-4" />;
    case '❌': return <AlertCircle className="h-4 w-4" />;
    case '👤': return <Users className="h-4 w-4" />;
    case '🔍': return <Activity className="h-4 w-4" />;
    default: return <Bell className="h-4 w-4" />;
  }
};

export default function SlackNotifications() {
  const [preferences, setPreferences] = useState<NotificationPreferences | null>(null);
  const [notificationTypes, setNotificationTypes] = useState<NotificationType[]>([]);
  const [slackStatus, setSlackStatus] = useState<SlackStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  
  // Configuration form state
  const [webhookUrl, setWebhookUrl] = useState('');
  const [defaultChannel, setDefaultChannel] = useState('');
  const [enabled, setEnabled] = useState(true);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [priorityFilter, setPriorityFilter] = useState('low');
  const [quietHoursStart, setQuietHoursStart] = useState<number | null>(null);
  const [quietHoursEnd, setQuietHoursEnd] = useState<number | null>(null);
  const [rateLimit, setRateLimit] = useState(60);
  const [mentionUsers, setMentionUsers] = useState<string[]>([]);
  const [testMessage, setTestMessage] = useState('This is a test notification from LLM Evaluation Platform');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      
      // Load preferences, notification types, and status in parallel
      const [prefsRes, typesRes, statusRes] = await Promise.all([
        fetch('/api/v1/slack/preferences'),
        fetch('/api/v1/slack/notification-types'),
        fetch('/api/v1/slack/status')
      ]);
      
      if (prefsRes.ok) {
        const prefsData = await prefsRes.json();
        setPreferences(prefsData);
        
        // Update form state
        setEnabled(prefsData.enabled);
        setDefaultChannel(prefsData.default_channel || '');
        setSelectedTypes(prefsData.notification_types || []);
        setPriorityFilter(prefsData.priority_filter || 'low');
        setQuietHoursStart(prefsData.quiet_hours_start);
        setQuietHoursEnd(prefsData.quiet_hours_end);
        setRateLimit(prefsData.rate_limit_per_hour || 60);
        setMentionUsers(prefsData.mention_users || []);
      }
      
      if (typesRes.ok) {
        const typesData = await typesRes.json();
        setNotificationTypes(typesData);
      }
      
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        setSlackStatus(statusData);
      }
      
    } catch (error) {
      console.error('Error loading Slack configuration:', error);
      setMessage({ type: 'error', text: 'Failed to load configuration' });
    } finally {
      setLoading(false);
    }
  };

  const handleQuickSetup = async () => {
    if (!webhookUrl) {
      setMessage({ type: 'error', text: 'Please enter a webhook URL' });
      return;
    }
    
    try {
      setSaving(true);
      
      const response = await fetch('/api/v1/slack/quick-setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          webhook_url: webhookUrl,
          channel: defaultChannel,
          enabled: enabled
        })
      });
      
      const result = await response.json();
      
      if (response.ok) {
        setMessage({ type: 'success', text: 'Slack integration configured successfully!' });
        await loadData(); // Refresh data
      } else {
        setMessage({ type: 'error', text: result.detail || 'Setup failed' });
      }
      
    } catch (error) {
      setMessage({ type: 'error', text: 'Error setting up Slack integration' });
    } finally {
      setSaving(false);
    }
  };

  const handleUpdatePreferences = async () => {
    try {
      setSaving(true);
      
      const response = await fetch('/api/v1/slack/preferences', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled,
          webhook_url: webhookUrl || null,
          default_channel: defaultChannel || null,
          notification_types: selectedTypes,
          priority_filter: priorityFilter,
          quiet_hours_start: quietHoursStart,
          quiet_hours_end: quietHoursEnd,
          rate_limit_per_hour: rateLimit,
          mention_users: mentionUsers
        })
      });
      
      const result = await response.json();
      
      if (response.ok) {
        setMessage({ type: 'success', text: 'Preferences updated successfully!' });
        await loadData(); // Refresh data
      } else {
        setMessage({ type: 'error', text: result.detail || 'Update failed' });
      }
      
    } catch (error) {
      setMessage({ type: 'error', text: 'Error updating preferences' });
    } finally {
      setSaving(false);
    }
  };

  const handleTestWebhook = async () => {
    try {
      setTesting(true);
      
      const response = await fetch('/api/v1/slack/test-webhook', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          webhook_url: webhookUrl || null,
          notification_type: 'alert',
          test_message: testMessage
        })
      });
      
      const result = await response.json();
      
      if (response.ok) {
        setMessage({ type: 'success', text: 'Test notification sent successfully!' });
      } else {
        setMessage({ type: 'error', text: result.detail || 'Test failed' });
      }
      
    } catch (error) {
      setMessage({ type: 'error', text: 'Error sending test notification' });
    } finally {
      setTesting(false);
    }
  };

  const toggleNotificationType = (type: string) => {
    setSelectedTypes(prev => 
      prev.includes(type) 
        ? prev.filter(t => t !== type)
        : [...prev, type]
    );
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="space-y-4">
          <div className="h-8 bg-gray-200 rounded animate-pulse"></div>
          <div className="h-64 bg-gray-200 rounded animate-pulse"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <MessageSquare className="h-8 w-8" />
            Slack Notifications
          </h1>
          <p className="text-gray-600 mt-1">
            Configure Slack integration for system alerts and notifications
          </p>
        </div>
        
        {slackStatus && (
          <div className="flex items-center gap-2">
            {getStatusIcon(slackStatus.webhook_status)}
            <span className="text-sm text-gray-600">
              {slackStatus.webhook_status === 'connected' ? 'Connected' : 
               slackStatus.webhook_status === 'not_configured' ? 'Not Configured' : 'Error'}
            </span>
          </div>
        )}
      </div>

      {message && (
        <Alert className={message.type === 'error' ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
          <AlertDescription className={message.type === 'error' ? 'text-red-700' : 'text-green-700'}>
            {message.text}
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="setup" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="setup">Setup</TabsTrigger>
          <TabsTrigger value="preferences">Preferences</TabsTrigger>
          <TabsTrigger value="test">Test</TabsTrigger>
          <TabsTrigger value="status">Status</TabsTrigger>
        </TabsList>

        <TabsContent value="setup" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Quick Setup
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="webhook-url">Slack Webhook URL</Label>
                <Input
                  id="webhook-url"
                  placeholder="https://hooks.slack.com/services/..."
                  value={webhookUrl}
                  onChange={(e) => setWebhookUrl(e.target.value)}
                />
                <p className="text-sm text-gray-500">
                  Create a webhook URL in your Slack workspace settings
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="default-channel">Default Channel (Optional)</Label>
                <Input
                  id="default-channel"
                  placeholder="#general"
                  value={defaultChannel}
                  onChange={(e) => setDefaultChannel(e.target.value)}
                />
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  id="enable-notifications"
                  checked={enabled}
                  onCheckedChange={setEnabled}
                />
                <Label htmlFor="enable-notifications">Enable notifications</Label>
              </div>

              <Button 
                onClick={handleQuickSetup}
                disabled={saving || !webhookUrl}
                className="w-full"
              >
                {saving ? 'Setting up...' : 'Setup Slack Integration'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="preferences" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Notification Types</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {notificationTypes.map((type) => (
                  <div 
                    key={type.type}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedTypes.includes(type.type) 
                        ? 'border-blue-500 bg-blue-50' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => toggleNotificationType(type.type)}
                  >
                    <div className="flex items-center gap-3">
                      {getNotificationIcon(type.icon)}
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <h3 className="font-medium">{type.name}</h3>
                          <Badge variant="outline" className={PRIORITY_LEVELS.find(p => p.value === type.priority)?.color}>
                            {type.priority}
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-600">{type.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Priority & Rate Limiting</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Minimum Priority</Label>
                  <Select value={priorityFilter} onValueChange={setPriorityFilter}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {PRIORITY_LEVELS.map((level) => (
                        <SelectItem key={level.value} value={level.value}>
                          {level.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="rate-limit">Rate Limit (per hour)</Label>
                  <Input
                    id="rate-limit"
                    type="number"
                    min="1"
                    max="1000"
                    value={rateLimit}
                    onChange={(e) => setRateLimit(parseInt(e.target.value) || 60)}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quiet Hours</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Start Hour (UTC)</Label>
                    <Select 
                      value={quietHoursStart?.toString() || ''} 
                      onValueChange={(value) => setQuietHoursStart(value ? parseInt(value) : null)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="None" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 24 }, (_, i) => (
                          <SelectItem key={i} value={i.toString()}>
                            {i.toString().padStart(2, '0')}:00
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>End Hour (UTC)</Label>
                    <Select 
                      value={quietHoursEnd?.toString() || ''} 
                      onValueChange={(value) => setQuietHoursEnd(value ? parseInt(value) : null)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="None" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 24 }, (_, i) => (
                          <SelectItem key={i} value={i.toString()}>
                            {i.toString().padStart(2, '0')}:00
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <p className="text-sm text-gray-500">
                  Notifications will be suppressed during these hours
                </p>
              </CardContent>
            </Card>
          </div>

          <Button 
            onClick={handleUpdatePreferences}
            disabled={saving}
            className="w-full"
          >
            {saving ? 'Saving...' : 'Save Preferences'}
          </Button>
        </TabsContent>

        <TabsContent value="test" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Send className="h-5 w-5" />
                Test Notifications
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="test-message">Test Message</Label>
                <Textarea
                  id="test-message"
                  value={testMessage}
                  onChange={(e) => setTestMessage(e.target.value)}
                  rows={3}
                />
              </div>

              <Button 
                onClick={handleTestWebhook}
                disabled={testing || !preferences?.webhook_configured}
                className="w-full"
              >
                {testing ? 'Sending...' : 'Send Test Notification'}
              </Button>

              {!preferences?.webhook_configured && (
                <p className="text-sm text-gray-500 text-center">
                  Configure a webhook URL first to test notifications
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="status" className="space-y-6">
          {slackStatus && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Integration Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {slackStatus.integration_enabled ? (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-gray-400" />
                    )}
                    <span className="font-medium">
                      {slackStatus.integration_enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Webhook Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(slackStatus.webhook_status)}
                    <span className="font-medium capitalize">
                      {slackStatus.webhook_status.replace('_', ' ')}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Notification Types</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {slackStatus.notification_types_count}/{slackStatus.total_notification_types}
                  </div>
                  <p className="text-sm text-gray-600">types enabled</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Rate Limit</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{slackStatus.rate_limit_per_hour}</div>
                  <p className="text-sm text-gray-600">per hour</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Quiet Hours</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {slackStatus.quiet_hours_configured ? (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    ) : (
                      <Clock className="h-5 w-5 text-gray-400" />
                    )}
                    <span className="font-medium">
                      {slackStatus.quiet_hours_configured ? 'Configured' : 'Not Set'}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Mentions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{slackStatus.mentions_configured}</div>
                  <p className="text-sm text-gray-600">users configured</p>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
} 