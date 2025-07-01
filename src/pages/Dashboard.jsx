import React from "react";
import DashboardPanel from "../components/DashboardPanel";

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-6">AI Trading Dashboard (Mock UI)</h1>
      <DashboardPanel />
    </div>
  );
}
