// src/components/ChartDisplay.jsx
import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

const mockPriceData = {
  labels: [
    "2025-06-20",
    "2025-06-21",
    "2025-06-22",
    "2025-06-23",
    "2025-06-24",
    "2025-06-25",
    "2025-06-26",
  ],
  datasets: [
    {
      label: "Price",
      data: [100, 102, 101, 105, 107, 110, 108],
      borderColor: "rgb(75, 192, 192)",
      backgroundColor: "rgba(75, 192, 192, 0.2)",
      tension: 0.3,
      fill: true,
    },
  ],
};

// Mock Mother AI summary stats
const mockMotherAIStats = {
  totalTrades: 15,
  winRate: 80, // %
  totalProfitPercent: 24.5, // %
};

export default function ChartDisplay({ symbol = "BTCUSDT" }) {
  return (
    <div className="w-full max-w-3xl mx-auto p-4 border rounded shadow">
      <h3 className="text-lg font-semibold mb-2">Chart: {symbol}</h3>
      <Line data={mockPriceData} />
      <div className="mt-4 p-4 bg-gray-100 rounded">
        <h4 className="font-semibold mb-2">Mother AI Summary</h4>
        <p>Total Trades: {mockMotherAIStats.totalTrades}</p>
        <p>Win Rate: {mockMotherAIStats.winRate}%</p>
        <p>Total Profit: {mockMotherAIStats.totalProfitPercent}%</p>
      </div>
    </div>
  );
}
