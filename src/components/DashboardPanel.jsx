import React, { Suspense } from "react";
import DashboardPanel from "../components/DashboardPanel";
import ChartDisplay from "../components/ChartDisplay";
import AgentPredictionCard from "../components/AgentPredictionCard";
// import MotherAIDecisionCard from "../components/MotherAIDecisionCard"; // Removed as requested earlier

export default function Dashboard() {
  const mockBalanceHistory = [
    { time: "2025-06-01", value: 1000 },
    { time: "2025-06-05", value: 1030 },
    { time: "2025-06-10", value: 1075 },
    { time: "2025-06-15", value: 1050 },
    { time: "2025-06-20", value: 1100 },
  ];

  return (
    <div className="space-y-8 p-6 md:p-10 bg-gray-900 min-h-screen text-white">
      <h1 className="text-4xl font-extrabold mb-6">AI Trading Dashboard</h1>

      {/* Portfolio Chart */}
      <ErrorBoundary fallback={<p className="text-red-500">Chart failed to load.</p>}>
        <ChartDisplay balanceHistory={mockBalanceHistory} />
      </ErrorBoundary>

      {/* Predictions Section */}
      <div className="grid md:grid-cols-2 gap-8">
        <ErrorBoundary fallback={<p className="text-red-500">Agent predictions unavailable.</p>}>
          <AgentPredictionCard />
        </ErrorBoundary>
        {/* <MotherAIDecisionCard /> */}
      </div>

      {/* Trade History / Insights Panel */}
      <ErrorBoundary fallback={<p className="text-red-500">Dashboard panel failed to load.</p>}>
        <DashboardPanel />
      </ErrorBoundary>
    </div>
  );
}

// Reusable local error boundary
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error("Dashboard error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || <p className="text-red-500">An error occurred.</p>;
    }

    return this.props.children;
  }
}
