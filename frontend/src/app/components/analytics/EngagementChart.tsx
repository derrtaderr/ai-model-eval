import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { UserEngagement } from '../../types/analytics';

interface EngagementChartProps {
  data: UserEngagement;
  loading?: boolean;
}

export function EngagementChart({ data, loading }: EngagementChartProps) {
  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center">
        <div className="animate-pulse text-gray-400">Loading engagement data...</div>
      </div>
    );
  }

  // Transform data for chart display
  const chartData = data.engagement_trends.daily_active_users.map((item, index) => ({
    date: new Date(item.date).toLocaleDateString(),
    activeUsers: item.count,
    evaluations: data.engagement_trends.daily_evaluations[index]?.count || 0
  }));

  if (chartData.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center">
        <div className="text-center">
          <div className="text-gray-400 text-sm">No engagement data available</div>
          <div className="text-gray-500 text-xs mt-1">Data will appear as users become active</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Key metrics summary */}
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">{data.user_metrics.active_users}</div>
          <div className="text-sm text-gray-600">Active Users</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">{data.user_metrics.new_users}</div>
          <div className="text-sm text-gray-600">New Users</div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="date" 
              stroke="#666"
              fontSize={12}
            />
            <YAxis 
              stroke="#666"
              fontSize={12}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '6px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
              }}
            />
            <Bar dataKey="activeUsers" fill="#3b82f6" name="Active Users" />
            <Bar dataKey="evaluations" fill="#10b981" name="Evaluations" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
} 