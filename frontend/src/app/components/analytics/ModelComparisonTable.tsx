import { ModelComparison } from '../../types/analytics';

interface ModelComparisonTableProps {
  data: ModelComparison[];
  loading?: boolean;
}

export function ModelComparisonTable({ data, loading }: ModelComparisonTableProps) {
  if (loading) {
    return (
      <div className="h-48 flex items-center justify-center">
        <div className="animate-pulse text-gray-400">Loading model comparison data...</div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center">
        <div className="text-center">
          <div className="text-gray-400 text-sm">No model comparison data available</div>
          <div className="text-gray-500 text-xs mt-1">Data will appear as models are evaluated</div>
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Model
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Evaluations
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Success Rate
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Avg Score
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Latency (ms)
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Cost per Eval
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Throughput/h
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((model, index) => (
            <tr key={model.model_name} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm font-medium text-gray-900">{model.model_name}</div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-gray-900">{model.total_evaluations.toLocaleString()}</div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="flex items-center">
                  <div className={`text-sm font-medium ${
                    model.success_rate > 0.9 ? 'text-green-600' : 
                    model.success_rate > 0.8 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {(model.success_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-gray-900">
                  {model.average_score.toFixed(2)} ± {model.score_std_dev.toFixed(2)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className={`text-sm ${
                  model.average_latency_ms < 1000 ? 'text-green-600' : 
                  model.average_latency_ms < 3000 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {model.average_latency_ms.toFixed(0)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-gray-900">
                  ${model.cost_per_evaluation.toFixed(4)}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-gray-900">
                  {model.throughput_per_hour.toFixed(1)}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Summary stats */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="text-lg font-bold text-blue-600">
            {data.reduce((sum, model) => sum + model.total_evaluations, 0).toLocaleString()}
          </div>
          <div className="text-sm text-gray-600">Total Evaluations</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-600">
            {((data.reduce((sum, model) => sum + model.success_rate, 0) / data.length) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Avg Success Rate</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-purple-600">
            {(data.reduce((sum, model) => sum + model.average_latency_ms, 0) / data.length).toFixed(0)}ms
          </div>
          <div className="text-sm text-gray-600">Avg Latency</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-orange-600">
            ${data.reduce((sum, model) => sum + model.total_cost_usd, 0).toFixed(2)}
          </div>
          <div className="text-sm text-gray-600">Total Cost</div>
        </div>
      </div>
    </div>
  );
} 