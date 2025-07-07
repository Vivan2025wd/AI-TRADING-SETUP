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
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshToggle, setRefreshToggle] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        // Fetch all trades combined from backend
        const res = await fetch(`/api/mother-ai/trades`);
        if (!res.ok) throw new Error(`Failed to load trade logs`);
        const data = await res.json();
        if (!isMounted) return;

        // Sort by timestamp ascending
        const sorted = [...data.data].sort(
          (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );

        let buyCount = 0;
        let sellCount = 0;

        const formatted = sorted.map((trade) => {
          const type = (trade.signal || "").toUpperCase(); // 'signal' field used here
          if (type === "BUY") buyCount++;
          else if (type === "SELL") sellCount++;

          return {
            time: new Date(trade.timestamp).toLocaleString(),
            rawTime: new Date(trade.timestamp).getTime(),
            type,
            price: trade.price || 0,
            value: trade.balance || 0,
            profit: trade.profit_percent ?? null,
            symbol: trade.symbol || "N/A",    // Include symbol from backend log
            sequenceLabel:
              type === "BUY"
                ? `Buy #${buyCount}`
                : type === "SELL"
                ? `Sell #${sellCount}`
                : "Balance",
          };
        });

        setHistory(formatted);
      } catch (err) {
        if (!isMounted) return;
        setError(err.message);
        setHistory([]);
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    fetchData();

    return () => {
      isMounted = false;
    };
  }, [refreshToggle]);

  const sortedData = [...history].sort((a, b) => a.rawTime - b.rawTime);
  const labels = sortedData.map((d) => d.time);

  const balanceDataset = {
    label: "Balance",
    data: sortedData.map((d) => d.value),
    borderColor: "rgb(34,197,94)",
    backgroundColor: "rgba(34,197,94,0.2)",
    tension: 0.4,
    fill: true,
    spanGaps: true,
  };

  const buyPoints = sortedData
    .map((d, i) =>
      d.type === "BUY"
        ? {
            x: labels[i],
            y: d.value,
            sequenceLabel: d.sequenceLabel,
            price: d.price,
            profit: d.profit,
          }
        : null
    )
    .filter(Boolean);

  const sellPoints = sortedData
    .map((d, i) =>
      d.type === "SELL"
        ? {
            x: labels[i],
            y: d.value,
            sequenceLabel: d.sequenceLabel,
            price: d.price,
            profit: d.profit,
          }
        : null
    )
    .filter(Boolean);

  const chartData = {
    labels,
    datasets: [
      balanceDataset,
      {
        label: "Buy",
        data: buyPoints,
        backgroundColor: "blue",
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: false,
      },
      {
        label: "Sell",
        data: sellPoints,
        backgroundColor: "red",
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: false,
      },
    ],
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
            const point = ctx.raw;
            if (ctx.dataset.label === "Buy" || ctx.dataset.label === "Sell") {
              return `${point.sequenceLabel}\nPrice: $${point.price}\nProfit: ${
                point.profit ?? "?"
              }%\nBalance: $${point.y}`;
            }
            return `Balance: $${ctx.parsed.y.toFixed(2)}`;
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
      <h2 className="text-white text-xl font-semibold">Performance for All Symbols</h2>

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

      {/* Table */}
      <div className="overflow-x-auto mt-4">
        <table className="min-w-full text-white bg-gray-800 rounded">
          <thead>
            <tr>
              <th className="py-2 px-4 border-b">Timestamp</th>
              <th className="py-2 px-4 border-b">Symbol</th> {/* New column */}
              <th className="py-2 px-4 border-b">Type</th>
              <th className="py-2 px-4 border-b">Price</th>
              <th className="py-2 px-4 border-b">Balance</th>
              <th className="py-2 px-4 border-b">Profit %</th>
            </tr>
          </thead>
          <tbody>
            {sortedData.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-4">
                  No data
                </td>
              </tr>
            ) : (
              sortedData.map((d, idx) => (
                <tr key={idx} className="hover:bg-gray-700">
                  <td className="py-2 px-4 border-b">{d.time}</td>
                  <td className="py-2 px-4 border-b">{d.symbol}</td>
                  <td className="py-2 px-4 border-b">{d.sequenceLabel}</td>
                  <td className="py-2 px-4 border-b">${d.price}</td>
                  <td className="py-2 px-4 border-b">${d.value.toFixed(2)}</td>
                  <td className="py-2 px-4 border-b">
                    {typeof d.profit === "number" ? `${d.profit.toFixed(2)}%` : "N/A"}
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
