import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Title, Tooltip, Legend, Filler);

export default function ChartDisplay({ balanceHistory }) {
  const formattedLabels = balanceHistory.map((point) =>
    new Date(point.time).toLocaleDateString("en-US", { month: "short", day: "numeric" })
  );

  const chartData = {
    labels: formattedLabels,
    datasets: [
      {
        label: "Balance Over Time",
        data: balanceHistory.map((point) => point.value),
        fill: true,
        borderColor: "#2563eb", // blue-600
        backgroundColor: (context) => {
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, 0, 0, 300);
          gradient.addColorStop(0, "rgba(37, 99, 235, 0.4)");
          gradient.addColorStop(1, "rgba(37, 99, 235, 0.05)");
          return gradient;
        },
        pointRadius: 3,
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { 
        position: "bottom", 
        labels: { color: "white" } 
      },
      title: {
        display: true,
        text: "Profit Curve by Mother AI",
        font: { size: 18 },
        color: "white",
      },
      tooltip: {
        backgroundColor: "#111827", // gray-900
        titleColor: "white",
        bodyColor: "white",
        callbacks: {
          label: (context) => `$${context.parsed.y.toFixed(2)}`,
        },
      },
    },
    scales: {
      y: {
        ticks: {
          color: "white",
          callback: (value) => `$${value}`,
        },
        title: { display: true, text: "Balance ($)", color: "white" },
        grid: { color: "#374151" }, // gray-700 grid lines
      },
      x: {
        ticks: { color: "white" },
        title: { display: true, text: "Date", color: "white" },
        grid: { color: "#374151" },
      },
    },
  };

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg p-6 h-[400px]">
      <Line data={chartData} options={chartOptions} />
    </div>
  );
}
