import { CheckCircle, AlertTriangle, XCircle, Activity } from 'lucide-react';
import { SystemHealth } from '../../types/analytics';

interface SystemHealthIndicatorProps {
  health: SystemHealth;
  loading?: boolean;
  className?: string;
}

export function SystemHealthIndicator({ health, loading, className = '' }: SystemHealthIndicatorProps) {
  if (loading) {
    return (
      <div className={`bg-white p-4 rounded-lg border border-gray-200 ${className}`}>
        <div className="flex items-center gap-3">
          <div className="animate-pulse w-6 h-6 bg-gray-200 rounded-full"></div>
          <div className="animate-pulse text-gray-400">Checking system health...</div>
        </div>
      </div>
    );
  }

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
      case 'online':
      case 'connected':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'warning':
      case 'degraded':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'error':
      case 'offline':
      case 'disconnected':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Activity className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
      case 'online':
      case 'connected':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'warning':
      case 'degraded':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'error':
      case 'offline':
      case 'disconnected':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  };

  return (
    <div className={`bg-white p-4 rounded-lg border border-gray-200 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {getStatusIcon(health.status)}
          <div>
            <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
            <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(health.status)}`}>
              {health.status.charAt(0).toUpperCase() + health.status.slice(1)}
            </div>
          </div>
        </div>

        <div className="text-right">
          <div className="text-sm text-gray-600">Uptime</div>
          <div className="text-lg font-semibold text-gray-900">
            {formatUptime(health.uptime_seconds)}
          </div>
        </div>
      </div>

      {/* Detailed status grid */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div>
            <div className="text-xs text-gray-600">Database</div>
            <div className="flex items-center gap-1">
              {getStatusIcon(health.database_status)}
              <span className="text-sm font-medium text-gray-900">
                {health.database_status}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div>
            <div className="text-xs text-gray-600">Batch Processor</div>
            <div className="flex items-center gap-1">
              {getStatusIcon(health.batch_processor_status)}
              <span className="text-sm font-medium text-gray-900">
                {health.batch_processor_status}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div>
            <div className="text-xs text-gray-600">Calibration</div>
            <div className="flex items-center gap-1">
              {getStatusIcon(health.calibration_system_status)}
              <span className="text-sm font-medium text-gray-900">
                {health.calibration_system_status}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Resource usage */}
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <div className="text-xs text-gray-600 mb-1">Memory Usage</div>
          <div className="text-sm font-medium text-gray-900">{health.memory_usage}</div>
        </div>
        <div>
          <div className="text-xs text-gray-600 mb-1">CPU Usage</div>
          <div className="text-sm font-medium text-gray-900">{health.cpu_usage}</div>
        </div>
      </div>
    </div>
  );
} 