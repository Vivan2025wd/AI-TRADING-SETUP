import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

export default function MotherAIBalanceChart() {
  const [symbols, setSymbols] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [refreshToggle, setRefreshToggle] = useState(false);

  const initialCapital = 1000; // <-- Set your starting capital here

  // Fetch available symbols
  useEffect(() => {
    async function fetchSymbols() {
      try {
        const res = await fetch("/api/mother-ai/trades");
        if (!res.ok) throw new Error("Failed to load symbols");

        const data = await res.json();
        const uniqueSymbols = Array.from(
          new Set(data.data.map((t) => t.symbol).filter(Boolean))
        );

        setSymbols(uniqueSymbols);
        if (uniqueSymbols.length > 0 && !selectedSymbol) {
          setSelectedSymbol(uniqueSymbols[0]);
        }
      } catch (err) {
        console.error(err);
        setError("Error loading symbols");
      }
    }

    fetchSymbols();
  }, []);

  // Fetch profit summary and build capital growth history
  useEffect(() => {
    if (!selectedSymbol) return;

    let isMounted = true;
    async function fetchProfitSummary() {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`/api/mother-ai/profits/${selectedSymbol}`);
        if (!res.ok) throw new Error(`No profit summary for ${selectedSymbol}`);

        const data = await res.json();
        if (!isMounted) return;

        const trades = Array.isArray(data.trades) ? data.trades : [];
        const sorted = [...trades].sort(
          (a, b) => new Date(a.exit_time) - new Date(b.exit_time)
        );

        let capital = initialCapital;
        const historyData = sorted.map((trade) => {
          capital += trade.pnl_dollars; // <-- Fix 1: Use pnl_dollars
          return {
            timestamp: trade.exit_time,
            balance: capital,
            profit_percent: ((capital - initialCapital) / initialCapital) * 100, // <-- Fix 2
          };
        });

        setHistory(historyData);
      } catch (err) {
        if (!isMounted) return;
        console.error(err);
        setError(err.message);
        setHistory([]);
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    fetchProfitSummary();

    return () => {
      isMounted = false;
    };
  }, [selectedSymbol, refreshToggle]);

  const labels = history.map((d) => new Date(d.timestamp).toLocaleString());
  const balanceDataset = {
    label: "Net Capital ($)",
    data: history.map((d) => d.balance),
    borderColor: "rgb(34,197,94)",
    backgroundColor: "rgba(34,197,94,0.2)",
    tension: 0.4,
    fill: true,
    spanGaps: true,
  };

  const chartData = {
    labels,
    datasets: [balanceDataset],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: true },
    plugins: {
      legend: { labels: { color: "white" } },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const d = history[ctx.dataIndex];
            if (!d) return "";
            return [
              `Balance: $${(d.balance ?? 0).toFixed(2)}`,
              `Profit %: ${
                typeof d.profit_percent === "number"
                  ? `${d.profit_percent.toFixed(2)}%`
                  : "N/A"
              }`,
              `Time: ${new Date(d.timestamp).toLocaleString()}`,
            ];
          },
        },
      },
    },
    scales: {
      x: { ticks: { color: "white" }, grid: { color: "#374151" } },
      y: { ticks: { color: "white" }, grid: { color: "#374151" } },
    },
  };

  return (
    <div className="space-y-4">
      <h2 className="text-white text-xl font-semibold">
        Net Capital Growth for{" "}
        <select
          className="bg-gray-700 text-white rounded px-2 py-1 ml-2"
          value={selectedSymbol}
          onChange={(e) => setSelectedSymbol(e.target.value)}
        >
          {symbols.map((sym) => (
            <option key={sym} value={sym}>
              {sym}
            </option>
          ))}
        </select>
      </h2>

      <div style={{ height: 300 }}>
        {loading && <p className="text-blue-400">Loading...</p>}
        {error && <p className="text-red-500">Error: {error}</p>}
        {!loading && !error && <Line data={chartData} options={options} />}
      </div>

      <button
        onClick={() => setRefreshToggle((v) => !v)}
        className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600"
      >
        Refresh
      </button>

      <div className="overflow-x-auto mt-4">
        <table className="min-w-full text-white bg-gray-800 rounded">
          <thead>
            <tr>
              <th className="py-2 px-4 border-b">Timestamp</th>
              <th className="py-2 px-4 border-b">Balance</th>
              <th className="py-2 px-4 border-b">Profit %</th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 ? (
              <tr>
                <td colSpan={3} className="text-center py-4">
                  No data
                </td>
              </tr>
            ) : (
              history.map((d, idx) => (
                <tr key={idx} className="hover:bg-gray-700">
                  <td className="py-2 px-4 border-b">
                    {new Date(d.timestamp).toLocaleString()}
                  </td>
                  <td className="py-2 px-4 border-b">${(d.balance ?? 0).toFixed(2)}</td>
                  <td className="py-2 px-4 border-b">
                    {typeof d.profit_percent === "number"
                      ? `${d.profit_percent.toFixed(2)}%`
                      : "N/A"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
