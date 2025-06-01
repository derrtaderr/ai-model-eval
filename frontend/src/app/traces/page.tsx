import { Database, Search, Filter, BarChart3, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function TracesPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center py-6">
            <Link href="/" className="flex items-center text-gray-600 hover:text-gray-900 mr-4">
              <ArrowLeft className="h-5 w-5 mr-1" />
              Back
            </Link>
            <Database className="h-8 w-8 text-blue-600 mr-3" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Trace Logging System</h1>
              <p className="text-sm text-gray-600">Automatic LLM interaction capture and analysis</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Database className="h-8 w-8 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Traces</p>
                <p className="text-2xl font-semibold text-gray-900">1,247</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <BarChart3 className="h-8 w-8 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Avg Response Time</p>
                <p className="text-2xl font-semibold text-gray-900">145ms</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Filter className="h-8 w-8 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Models Used</p>
                <p className="text-2xl font-semibold text-gray-900">5</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Search className="h-8 w-8 text-orange-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Today&apos;s Traces</p>
                <p className="text-2xl font-semibold text-gray-900">89</p>
              </div>
            </div>
          </div>
        </div>

        {/* Features Showcase */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">🔄 LangSmith Integration</h3>
            <ul className="space-y-2 text-gray-600">
              <li>✅ Automatic trace capture</li>
              <li>✅ Full context preservation (prompts, outputs, tool calls)</li>
              <li>✅ Real-time synchronization</li>
              <li>✅ Custom metadata enrichment</li>
            </ul>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">🔍 Advanced Filtering</h3>
            <ul className="space-y-2 text-gray-600">
              <li>✅ Multi-dimensional search</li>
              <li>✅ Model and provider filtering</li>
              <li>✅ Date range queries</li>
              <li>✅ Performance-based filtering</li>
            </ul>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">📊 Auto-Tagging System</h3>
            <ul className="space-y-2 text-gray-600">
              <li>✅ Provider-based categorization</li>
              <li>✅ Latency bucketing</li>
              <li>✅ Tool usage tracking</li>
              <li>✅ Custom taxonomy support</li>
            </ul>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">🚀 REST API</h3>
            <ul className="space-y-2 text-gray-600">
              <li>✅ Complete CRUD operations</li>
              <li>✅ Batch operations</li>
              <li>✅ Statistics endpoints</li>
              <li>✅ Export capabilities</li>
            </ul>
          </div>
        </div>

        {/* API Demo */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">🛠️ Available API Endpoints</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">GET /api/traces</span>
                <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">List all traces</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">POST /api/traces</span>
                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">Create trace</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">POST /api/traces/search</span>
                <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">Advanced search</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">POST /api/traces/sync-langsmith</span>
                <span className="text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded">Sync data</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">GET /api/traces/stats/summary</span>
                <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">Statistics</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">GET /api/traces/{`{id}`}</span>
                <span className="text-xs bg-gray-100 text-gray-800 px-2 py-1 rounded">Get trace</span>
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <p className="text-blue-800 text-sm">
              💡 <strong>Try it out:</strong> Visit{" "}
              <a href="http://localhost:8000/docs" target="_blank" className="underline font-medium">
                http://localhost:8000/docs
              </a>{" "}
              to explore the interactive API documentation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
} 