import React from "react";
import DashboardPanel from "../components/DashboardPanel";
import ChartDisplay from "../components/ChartDisplay";
import AgentPredictionCard from "../components/AgentPredictionCard";
import MotherAIDecisionCard from "../components/MotherAIDecisionCard";

export default function Dashboard() {
  return (
    <div>
      <DashboardPanel />
      <ChartDisplay />
      <AgentPredictionCard />
      <MotherAIDecisionCard />
    </div>
  );
}
