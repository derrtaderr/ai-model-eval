import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { AgreementAnalysis } from '../../types/analytics';

interface AgreementChartProps {
  data: AgreementAnalysis;
  loading?: boolean;
}

export function AgreementChart({ data, loading }: AgreementChartProps) {
  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center">
        <div className="animate-pulse text-gray-400">Loading agreement analysis...</div>
      </div>
    );
  }

  // Prepare data for pie chart
  const agreementData = [
    {
      name: 'Strong Agreement',
      value: Math.round(data.strong_agreement_rate * 100),
      color: '#10b981'
    },
    {
      name: 'General Agreement',
      value: Math.round((data.agreement_rate - data.strong_agreement_rate) * 100),
      color: '#3b82f6'
    },
    {
      name: 'Disagreement',
      value: Math.round((1 - data.agreement_rate) * 100),
      color: '#ef4444'
    }
  ];

  const COLORS = ['#10b981', '#3b82f6', '#ef4444'];

  return (
    <div className="space-y-4">
      {/* Key metrics */}
      <div className="grid grid-cols-2 gap-4 text-center">
        <div>
          <div className="text-lg font-bold text-gray-900">
            {(data.agreement_rate * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Overall Agreement</div>
        </div>
        <div>
          <div className="text-lg font-bold text-gray-900">
            {data.total_comparisons}
          </div>
          <div className="text-sm text-gray-600">Comparisons</div>
        </div>
      </div>

      {/* Pie chart */}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={agreementData}
              cx="50%"
              cy="50%"
              innerRadius={40}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
            >
              {agreementData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip 
              formatter={(value) => [`${value}%`, 'Agreement Level']}
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '6px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
              }}
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Disagreement patterns */}
      <div className="text-sm">
        <div className="font-medium text-gray-900 mb-2">Disagreement Patterns:</div>
        <div className="space-y-1">
          <div className="flex justify-between">
            <span className="text-gray-600">AI Higher Scores:</span>
            <span className="text-gray-900">{data.disagreement_patterns.ai_higher}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Human Higher Scores:</span>
            <span className="text-gray-900">{data.disagreement_patterns.human_higher}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Avg Disagreement:</span>
            <span className="text-gray-900">{data.disagreement_patterns.avg_disagreement.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
} 