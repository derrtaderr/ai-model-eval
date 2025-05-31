import { Activity, TestTube, Users, BarChart3, Database, Zap } from "lucide-react";

interface FeatureCard {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  status: "completed" | "in-progress" | "planned";
  href?: string;
}

const features: FeatureCard[] = [
  {
    title: "Trace Logging",
    description: "Automatic LLM interaction capture with LangSmith integration",
    icon: Database,
    status: "completed",
    href: "/traces"
  },
  {
    title: "Unit Testing Framework",
    description: "7 assertion types for comprehensive LLM output validation",
    icon: TestTube,
    status: "completed",
    href: "/tests"
  },
  {
    title: "Human Evaluation Dashboard",
    description: "Systematic review interface for qualitative assessment", 
    icon: Users,
    status: "in-progress",
    href: "/evaluation"
  },
  {
    title: "A/B Testing Framework",
    description: "Experiment management for measuring product impact",
    icon: BarChart3,
    status: "planned"
  },
  {
    title: "Analytics Engine",
    description: "Comprehensive metrics tracking and performance monitoring",
    icon: Activity,
    status: "planned"
  },
  {
    title: "Model-Based Evaluation",
    description: "LLM-powered automatic evaluation at scale",
    icon: Zap,
    status: "planned"
  }
];

function StatusBadge({ status }: { status: FeatureCard["status"] }) {
  const styles = {
    completed: "bg-green-100 text-green-800 border-green-200",
    "in-progress": "bg-blue-100 text-blue-800 border-blue-200",
    planned: "bg-gray-100 text-gray-600 border-gray-200"
  };
  
  const labels = {
    completed: "✅ Complete",
    "in-progress": "🚧 In Progress", 
    planned: "📋 Planned"
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${styles[status]}`}>
      {labels[status]}
    </span>
  );
}

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <TestTube className="h-8 w-8 text-blue-600 mr-3" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">LLM Evaluation Platform</h1>
                <p className="text-sm text-gray-600">Three-tier evaluation system for LLM-powered products</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                Progress: <span className="font-semibold text-blue-600">3/12 Tasks Complete (25%)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Three-Tier System Overview */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">Three-Tier Evaluation System</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-green-500">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Level 1: Unit Tests</h3>
              <p className="text-gray-600 mb-3">Automated assertions and functional tests</p>
              <StatusBadge status="completed" />
            </div>
            <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-blue-500">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Level 2: Human Evaluation</h3>
              <p className="text-gray-600 mb-3">Qualitative review with labeling</p>
              <StatusBadge status="in-progress" />
            </div>
            <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-purple-500">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Level 3: A/B Testing</h3>
              <p className="text-gray-600 mb-3">Product-level impact measurement</p>
              <StatusBadge status="planned" />
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Platform Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div key={index} className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <feature.icon className="h-8 w-8 text-blue-600" />
                    <StatusBadge status={feature.status} />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
                  <p className="text-gray-600 text-sm mb-4">{feature.description}</p>
                  {feature.href && feature.status === "completed" && (
                    <a 
                      href={feature.href}
                      className="inline-flex items-center text-blue-600 hover:text-blue-800 text-sm font-medium"
                    >
                      View Details →
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* API Status */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">System Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
                <span className="font-medium text-gray-900">Backend API</span>
              </div>
              <span className="text-green-600 text-sm">Running on :8000</span>
            </div>
            <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-blue-500 rounded-full mr-3"></div>
                <span className="font-medium text-gray-900">Frontend Dev Server</span>
              </div>
              <span className="text-blue-600 text-sm">Running on :3000</span>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-gray-600 text-sm">
              🎉 Development environment ready! Both frontend and backend servers are running.
            </p>
            <div className="mt-2">
              <a 
                href="http://localhost:8000/docs" 
                target="_blank"
                className="text-blue-600 hover:text-blue-800 text-sm mr-4"
              >
                API Documentation →
              </a>
              <a 
                href="http://localhost:8000/health" 
                target="_blank"
                className="text-green-600 hover:text-green-800 text-sm"
              >
                Health Check →
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
