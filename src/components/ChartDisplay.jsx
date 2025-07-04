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
  const [balanceHistory, setBalanceHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [limit, setLimit] = useState(100);
  const [refreshToggle, setRefreshToggle] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function fetchTradeHistory() {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`/api/mother-ai/trades?limit=${limit}`);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: Failed to fetch trade data`);
        }

        const data = await response.json();

        if (!data || !Array.isArray(data.data)) {
          throw new Error("Invalid data format received");
        }

        if (!isMounted) return;

        // Sort data here ascending by timestamp to assign sequence properly
        const sortedRaw = [...data.data].sort(
          (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );

        // Add trade sequence number for buy/sell separately
        let buyCount = 0;
        let sellCount = 0;
        const formattedData = sortedRaw.map((trade) => {
          let labelType = trade.type.toUpperCase();
          if (labelType === "BUY") buyCount += 1;
          else if (labelType === "SELL") sellCount += 1;

          return {
            time: trade.timestamp ? new Date(trade.timestamp).toLocaleString() : "Unknown",
            rawTime: trade.timestamp ? new Date(trade.timestamp).getTime() : 0,
            value: typeof trade.balance === "number" ? trade.balance : 0,
            type: labelType,
            agent: trade.agent || "Unknown Agent",
            price: trade.price || 0,
            result: trade.result || "N/A",
            sequenceLabel:
              labelType === "BUY"
                ? `Buy #${buyCount}`
                : labelType === "SELL"
                ? `Sell #${sellCount}`
                : labelType,
          };
        });

        setBalanceHistory(formattedData);
      } catch (err) {
        if (!isMounted) return;
        console.error("Fetch error:", err);
        setError(err.message || "Unknown error while fetching trade data");
        setBalanceHistory([]);
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    fetchTradeHistory();

    return () => {
      isMounted = false;
    };
  }, [limit, refreshToggle]);

  // Sort data ascending by rawTime for chart correctness
  const sortedData = [...balanceHistory].sort((a, b) => a.rawTime - b.rawTime);

  const labels = sortedData.map((d) => d.time);

  const balanceDataset = {
    label: "Balance Over Time",
    data: sortedData.map((d) => d.value),
    fill: true,
    borderColor: "rgb(34,197,94)",
    backgroundColor: "rgba(34,197,94,0.2)",
    tension: 0.3,
    spanGaps: true,
  };

  const buyPoints = sortedData
    .map((d, i) =>
      d.type === "BUY" ? { x: labels[i], y: d.value, agent: d.agent, sequenceLabel: d.sequenceLabel } : null
    )
    .filter(Boolean);

  const buyPointsDataset = {
    label: "Buy Trades",
    data: buyPoints.map((point) => ({
      x: point.x,
      y: point.y,
      agent: point.agent,
      sequenceLabel: point.sequenceLabel,
    })),
    showLine: false,
    backgroundColor: "blue",
    pointRadius: 6,
    pointHoverRadius: 8,
  };

  const data = {
    labels,
    datasets: [balanceDataset, buyPointsDataset],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "nearest",
      intersect: true,
    },
    plugins: {
      legend: { labels: { color: "white" } },
      tooltip: {
        callbacks: {
          label: (context) => {
            const datasetLabel = context.dataset.label || "";
            if (datasetLabel === "Buy Trades") {
              const agent = context.raw.agent || "Unknown Agent";
              const seqLabel = context.raw.sequenceLabel || "Buy";
              return `${seqLabel} by: ${agent}\nBalance: $${context.parsed.y.toFixed(2)}`;
            }
            return `${datasetLabel}: $${context.parsed.y.toFixed(2)}`;
          },
          title: (context) => context[0].label || "",
        },
        backgroundColor: "#111827",
        titleColor: "white",
        bodyColor: "white",
      },
    },
    scales: {
      x: {
        ticks: { color: "white" },
        grid: { color: "#374151" },
      },
      y: {
        ticks: { color: "white" },
        grid: { color: "#374151" },
        position: "left",
      },
    },
  };

  function loadMore() {
    setLimit((prev) => prev + 100);
  }

  function refresh() {
    setRefreshToggle((prev) => !prev);
  }

  return (
    <div className="space-y-4">
      <div style={{ height: 300 }}>
        {loading && <p className="text-blue-400">Loading balance chart...</p>}
        {error && <p className="text-red-500">Error: {error}</p>}
        <Line data={data} options={options} />
      </div>

      <div className="flex space-x-2">
        {!loading && !error && balanceHistory.length >= limit && (
          <button
            onClick={loadMore}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Load More
          </button>
        )}
        <button
          onClick={refresh}
          disabled={loading}
          className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
        >
          Refresh
        </button>
      </div>

      {/* Trade Logs Table */}
      <div className="overflow-x-auto mt-6">
        <table className="min-w-full bg-gray-800 text-white rounded">
          <thead>
            <tr>
              <th className="py-2 px-4 border-b border-gray-600">Timestamp</th>
              <th className="py-2 px-4 border-b border-gray-600">Type</th>
              <th className="py-2 px-4 border-b border-gray-600">Agent</th>
              <th className="py-2 px-4 border-b border-gray-600">Price</th>
              <th className="py-2 px-4 border-b border-gray-600">Balance</th>
              <th className="py-2 px-4 border-b border-gray-600">Result</th>
            </tr>
          </thead>
          <tbody>
            {sortedData.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-4">
                  No trade data available
                </td>
              </tr>
            ) : (
              sortedData.map((trade, idx) => (
                <tr key={idx} className="hover:bg-gray-700">
                  <td className="py-2 px-4 border-b border-gray-600">{trade.time}</td>
                  <td className="py-2 px-4 border-b border-gray-600">{trade.sequenceLabel}</td>
                  <td className="py-2 px-4 border-b border-gray-600">{trade.agent}</td>
                  <td className="py-2 px-4 border-b border-gray-600">${trade.price.toFixed(2)}</td>
                  <td className="py-2 px-4 border-b border-gray-600">${trade.value.toFixed(2)}</td>
                  <td className="py-2 px-4 border-b border-gray-600">{trade.result}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
