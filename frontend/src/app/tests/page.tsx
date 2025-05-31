import { TestTube, Zap, CheckCircle, XCircle, ArrowLeft, Play, Settings } from "lucide-react";

export default function TestsPage() {
  const assertionTypes = [
    { name: "contains", description: "Text content validation", icon: "📝", status: "active" },
    { name: "not_contains", description: "Safety and content filtering", icon: "🛡️", status: "active" },
    { name: "regex", description: "Pattern matching for structured outputs", icon: "🔍", status: "active" },
    { name: "sentiment", description: "Emotion and tone analysis", icon: "😊", status: "active" },
    { name: "json_schema", description: "Structured data validation", icon: "📋", status: "active" },
    { name: "length", description: "Response length constraints", icon: "📏", status: "active" },
    { name: "custom_function", description: "Business logic validation", icon: "⚙️", status: "active" }
  ];

  const testResults = [
    { id: 1, name: "Greeting Response Test", status: "passed", time: "45ms", assertions: 1 },
    { id: 2, name: "Safety Content Filter", status: "passed", time: "23ms", assertions: 1 },
    { id: 3, name: "JSON Schema Validation", status: "failed", time: "67ms", assertions: 1 },
    { id: 4, name: "Email Format Check", status: "passed", time: "31ms", assertions: 1 },
    { id: 5, name: "Sentiment Analysis", status: "passed", time: "89ms", assertions: 1 }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center py-6">
            <a href="/" className="flex items-center text-gray-600 hover:text-gray-900 mr-4">
              <ArrowLeft className="h-5 w-5 mr-1" />
              Back
            </a>
            <TestTube className="h-8 w-8 text-blue-600 mr-3" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Unit Testing Framework</h1>
              <p className="text-sm text-gray-600">Comprehensive LLM output validation system</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <TestTube className="h-8 w-8 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Test Cases</p>
                <p className="text-2xl font-semibold text-gray-900">156</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <CheckCircle className="h-8 w-8 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Pass Rate</p>
                <p className="text-2xl font-semibold text-gray-900">94.2%</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Zap className="h-8 w-8 text-yellow-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Avg Execution</p>
                <p className="text-2xl font-semibold text-gray-900">47ms</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Play className="h-8 w-8 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Tests Today</p>
                <p className="text-2xl font-semibold text-gray-900">1,203</p>
              </div>
            </div>
          </div>
        </div>

        {/* Assertion Types */}
        <div className="mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-6">🧪 Available Assertion Types</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {assertionTypes.map((assertion, index) => (
              <div key={index} className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center">
                    <span className="text-2xl mr-3">{assertion.icon}</span>
                    <span className="font-semibold text-gray-900">{assertion.name}</span>
                  </div>
                  <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">Active</span>
                </div>
                <p className="text-gray-600 text-sm">{assertion.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Framework Features */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">⚡ Performance Features</h3>
            <ul className="space-y-2 text-gray-600">
              <li>✅ Parallel test execution (4 workers)</li>
              <li>✅ Average execution time &lt;50ms</li>
              <li>✅ Concurrent test processing</li>
              <li>✅ Performance metrics tracking</li>
            </ul>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">🔄 Advanced Capabilities</h3>
            <ul className="space-y-2 text-gray-600">
              <li>✅ Regression testing</li>
              <li>✅ Test suite management</li>
              <li>✅ Baseline comparison</li>
              <li>✅ Automated test runs</li>
            </ul>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">🛡️ Safety & Validation</h3>
            <ul className="space-y-2 text-gray-600">
              <li>✅ Content filtering assertions</li>
              <li>✅ Safe code execution environment</li>
              <li>✅ Input sanitization</li>
              <li>✅ Error handling & recovery</li>
            </ul>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">📊 Management & API</h3>
            <ul className="space-y-2 text-gray-600">
              <li>✅ Complete REST API</li>
              <li>✅ Test case CRUD operations</li>
              <li>✅ Execution history tracking</li>
              <li>✅ Batch test execution</li>
            </ul>
          </div>
        </div>

        {/* Recent Test Results */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">📈 Recent Test Results</h3>
            <button className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              <Play className="h-4 w-4 mr-2" />
              Run Tests
            </button>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Test Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Execution Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Assertions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {testResults.map((test) => (
                  <tr key={test.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {test.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        test.status === 'passed' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {test.status === 'passed' ? (
                          <CheckCircle className="h-3 w-3 mr-1" />
                        ) : (
                          <XCircle className="h-3 w-3 mr-1" />
                        )}
                        {test.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {test.time}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {test.assertions}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* API Information */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">🛠️ Testing API Endpoints</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">POST /api/test-cases</span>
                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">Create test</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">POST /api/test-runs</span>
                <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">Run tests</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">GET /api/assertions/types</span>
                <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">List types</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">POST /api/test-runs/regression</span>
                <span className="text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded">Regression</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">GET /api/test-runs</span>
                <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">History</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">POST /api/test-runs/trace/{`{id}`}</span>
                <span className="text-xs bg-gray-100 text-gray-800 px-2 py-1 rounded">Test trace</span>
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <p className="text-blue-800 text-sm">
              🧪 <strong>Demo available:</strong> Run{" "}
              <code className="bg-blue-100 px-2 py-1 rounded font-mono text-xs">python test_framework_demo.py</code>{" "}
              in the project root to see all assertion types in action.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
} 