"use client";

import React, { useState } from "react";
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

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Line = dynamic(() => import("react-chartjs-2").then((mod) => mod.Line), {
  ssr: false,
});
import LoadingDots from "../../Components/ui/LoadingDots";

import simulateWithDrift from "../../../utils/brownianMotion/simulateWithDrift";
import simulateWithoutDrift from "../../../utils/brownianMotion/simulateWithoutDrift";
import simulateOrnsteinUhlenbeck from "../../../utils/brownianMotion/simulateOrnsteinUhlenbeck";
import simulateGeometricBrownianMotion from "../../../utils/brownianMotion/simulateGeometric";

const defaultInputs = {
  noDrift: {
    steps: 500,
    particles: 5,
    timeStep: 10,
    volatility: 0.2,
    initialValue: 0,
  },
  drift: {
    steps: 500,
    particles: 5,
    timeStep: 10,
    drift: 0.05,
    volatility: 0.2,
    initialValue: 0,
  },
  geometric: {
    drift: 0.05,
    volatility: 0.2,
    steps: 100,
    particles: 5,
    timeStep: 1,
    initialValue: 1,
  },
  ornstein: {
    steps: 500,
    particles: 5,
    timeStep: 1,
    theta: 0.7,
    mean: 0.0,
    volatility: 0.2,
    initialValue: 1,
  },
};

const inputLabels = {
  steps: "Number of Steps",
  particles: "Number of Paths",
  timeStep: "Time Step Size",
  drift: "Drift (μ)",
  theta: "Mean Reversion Speed (θ)",
  mean: "Mean Reversion Level (μ)",
  volatility: "Volatility (σ)",
  initialValue: "Initial Value",
};

function BrownianMotionPage() {
  const [application, setApplication] = useState(null);
  const [inputs, setInputs] = useState({});
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const applications = [
    { title: "Brownian Motion without Drift", key: "noDrift" },
    { title: "Brownian Motion with Drift", key: "drift" },
    { title: "Geometric Brownian Motion", key: "geometric" },
    { title: "Ornstein-Uhlenbeck Process", key: "ornstein" },
  ];

  const handleApplicationChange = (key) => {
    setApplication(key);
    setInputs(defaultInputs[key]);
    setResults(null);
  };

  const runSimulation = () => {
    setIsLoading(true);
    setTimeout(() => {
      let simulationResults;

      switch (application) {
        case "drift":
          simulationResults = simulateWithDrift(inputs);
          break;
        case "noDrift":
          simulationResults = simulateWithoutDrift(inputs);
          break;
        case "ornstein":
          simulationResults = simulateOrnsteinUhlenbeck(inputs);
          break;
        case "geometric":
          simulationResults = simulateGeometricBrownianMotion(inputs);
          break;
        default:
          simulationResults = null;
      }

      setResults(simulationResults);
      setIsLoading(false);
    }, 1000);
  };

  const buildChartData = (trajectories) => {
    if (!trajectories || trajectories.length === 0) return null;

    const maxLen = Math.max(...trajectories.map((t) => t.length));
    const labels = Array.from({ length: maxLen }, (_, i) => i);

    const datasets = trajectories.map((trajectory, idx) => ({
      label: `Path ${idx + 1}`,
      data: trajectory,
      borderColor: `hsl(${(idx * 360) / trajectories.length}, 70%, 50%)`,
      fill: false,
    }));

    return { labels, datasets };
  };

  const stats = results
    ? (() => {
        const finalValues = results.map((path) => path[path.length - 1]);
        const mean =
          finalValues.reduce((a, b) => a + b, 0) / finalValues.length;
        const stdDev = Math.sqrt(
          finalValues.reduce(
            (sum, value) => sum + Math.pow(value - mean, 2),
            0
          ) / finalValues.length
        );
        return { mean, stdDev };
      })()
    : null;

  const chartData = results ? buildChartData(results) : null;

  return (
    <div className="min-h-screen px-4 py-10 flex flex-col gap-10 bg-gray-50">
      <div className="text-center max-w-4xl mx-auto">
        <h1 className="text-3xl font-extrabold mb-4">
          Brownian Motion Simulations
        </h1>
        <p className="text-gray-600">
          Explore various stochastic processes based on Brownian motion.
        </p>
      </div>

      <div className="max-w-4xl mx-auto text-center mt-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {applications.map((app) => (
            <button
              key={app.key}
              onClick={() => handleApplicationChange(app.key)}
              className={`p-6 rounded-lg shadow-md transition transform hover:scale-105 ${
                application === app.key
                  ? "bg-blue-100 border-blue-500 border-2"
                  : "bg-white"
              }`}
            >
              <h3 className="text-lg font-bold">{app.title}</h3>
            </button>
          ))}
        </div>
      </div>

      {application && (
        <div className="flex flex-col md:flex-row gap-6">
          <div className="md:w-1/4 bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-bold mb-2">
              {application.charAt(0).toUpperCase() + application.slice(1)}
            </h3>
            <div className="space-y-4">
              {Object.entries(inputs).map(([key, value]) => (
                <div key={key}>
                  <label className="block mb-1 font-medium">
                    {inputLabels[key] || key}
                  </label>
                  <input
                    type="number"
                    className="border p-2 rounded w-full"
                    value={value}
                    onChange={(e) =>
                      setInputs((prev) => ({
                        ...prev,
                        [key]: parseFloat(e.target.value),
                      }))
                    }
                  />
                </div>
              ))}
            </div>
            <button
              onClick={runSimulation}
              disabled={isLoading}
              className="w-full py-2 mt-4 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              {isLoading ? <LoadingDots color="white" /> : "Run Simulation"}
            </button>
          </div>

          <div className="md:w-3/4 bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-bold mb-4">Results</h3>
            {isLoading ? (
              <div className="flex justify-center items-center h-64">
                <LoadingDots color="#3498db" />
              </div>
            ) : results ? (
              <>
                {chartData && (
                  <div className="h-[40rem]">
                    <Line
                      data={chartData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: { display: true },
                          title: {
                            display: true,
                            text: `Simulation: ${application.toUpperCase()}`,
                          },
                        },
                        scales: {
                          x: { title: { display: true, text: "Step" } },
                          y: { title: { display: true, text: "Value" } },
                        },
                        elements: {
                          point: { radius: 0 },
                        },
                      }}
                    />
                  </div>
                )}
                {stats && (
                  <div className="mt-4">
                    <p className="font-medium">
                      Mean Final Value: {stats.mean.toFixed(2)}
                    </p>
                    <p className="font-medium">
                      Std Dev of Final Value: {stats.stdDev.toFixed(2)}
                    </p>
                  </div>
                )}
              </>
            ) : (
              <p className="text-gray-500">
                Run the simulation to view results.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default BrownianMotionPage;
