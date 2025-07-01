import React from "react";
import DashboardPanel from "../components/DashboardPanel";
import ChartDisplay from "../components/ChartDisplay";
import AgentPredictionCard from "../components/AgentPredictionCard";
// import MotherAIDecisionCard from "../components/MotherAIDecisionCard"; // Removed as requested earlier

export default function Dashboard() {
  // Example mock data for chart (replace with real props if available)
  const mockBalanceHistory = [
    { time: "2025-06-01", value: 1000 },
    { time: "2025-06-05", value: 1030 },
    { time: "2025-06-10", value: 1075 },
    { time: "2025-06-15", value: 1050 },
    { time: "2025-06-20", value: 1100 },
  ];

  return (
    <div className="space-y-8 p-6 md:p-10 bg-gray-900 min-h-screen text-white">
      <h1 className="text-4xl font-extrabold text-white mb-6">AI Trading Dashboard</h1>

      {/* Portfolio chart */}
      <ChartDisplay balanceHistory={mockBalanceHistory} />

      {/* Predictions */}
      <div className="grid md:grid-cols-2 gap-8">
        <AgentPredictionCard />
        {/* <MotherAIDecisionCard /> — removed since you said only agent predictions */}
      </div>

      {/* Trade history / insights */}
      <DashboardPanel />
    </div>
  );
}
