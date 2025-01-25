"use client";

import React from "react";
import dynamic from "next/dynamic";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register required Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Dynamically import the Line component from react-chartjs-2
const Line = dynamic(() => import("react-chartjs-2").then((mod) => mod.Line), {
  ssr: false,
});
const BrownianLineChart = ({ trajectories, timeSteps }) => {
  const datasets = trajectories.map((trajectory, idx) => ({
    label: `Particle ${idx + 1}`,
    data: trajectory,
    borderColor: `rgba(75, 192, 192, ${0.6 + idx / trajectories.length})`,
    fill: false,
  }));

  const chartData = {
    labels: Array.from({ length: timeSteps + 1 }, (_, i) => i),
    datasets,
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true },
      title: { display: true, text: "Brownian Motion Simulation (1D)" },
    },
    scales: {
      x: { title: { display: true, text: "Time Steps" } },
      y: { title: { display: true, text: "Position" } },
    },
  };

  return (
    <div className="w-full h-96">
      <Line type="line" data={chartData} options={options} />
    </div>
  );
};

export default BrownianLineChart;
