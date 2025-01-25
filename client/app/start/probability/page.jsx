"use client";

import React, { useState } from "react";
import ProbabilityBarChart from "../../Components/Charts/ProbabilityBarChart";
import LoadingDots from "../../Components/ui/LoadingDots";

const distributions = [
  {
    type: "normal",
    name: "Normal Distribution",
    description: "Models continuous data with a bell-shaped curve.",
  },
  {
    type: "uniform",
    name: "Uniform Distribution",
    description: "Equal probability for all values in a range.",
  },
  {
    type: "exponential",
    name: "Exponential Distribution",
    description: "Models time between events in a Poisson process.",
  },
  {
    type: "poisson",
    name: "Poisson Distribution",
    description: "Probability of events in a fixed interval.",
  },
  {
    type: "binomial",
    name: "Binomial Distribution",
    description: "Counts successes in a fixed number of trials.",
  },
];

const defaultInputs = {
  normal: { mu: 0, sigma: 1, N: 1000 },
  uniform: { a: 0, b: 1, N: 1000 },
  exponential: { lambda: 1, N: 1000 },
  poisson: { lambda: 1, N: 1000 },
  binomial: { n: 10, p: 0.5, N: 1000 },
};

const inputLabels = {
  normal: {
    mu: "Mean (μ)",
    sigma: "Standard Deviation (σ)",
    N: "Number of Simulations (N)",
  },
  uniform: {
    a: "Start of Range (a)",
    b: "End of Range (b)",
    N: "Number of Simulations (N)",
  },
  exponential: {
    lambda: "Rate Parameter (λ)",
    N: "Number of Simulations (N)",
  },
  poisson: {
    lambda: "Rate Parameter (λ)",
    N: "Number of Simulations (N)",
  },
  binomial: {
    n: "Number of Trials (n)",
    p: "Probability of Success (p)",
    N: "Number of Simulations (N)",
  },
};

const ProbabilityDistributionsPage = () => {
  const [distributionType, setDistributionType] = useState(null);
  const [inputs, setInputs] = useState({});
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleDistributionChange = (type) => {
    setDistributionType(type);
    setInputs(defaultInputs[type]);
    setResults(null);
  };

  const handleInputChange = (key, value) => {
    setInputs((prev) => ({
      ...prev,
      [key]: parseFloat(value) || 0,
    }));
  };

  const simulateDistribution = async () => {
    setIsLoading(true);
    const simulate = await import(
      `../../../utils/probability/${distributionType}.js`
    );
    const simulationResults = simulate.default(inputs);
    setResults(simulationResults);
    setIsLoading(false);
  };

  return (
    <div className="h-fit px-4 lg:px-10 py-10 flex flex-col gap-10 bg-gray-50">
      <div className="text-center max-w-4xl mx-auto">
        <h1 className="text-3xl font-extrabold mb-4">
          Probability Distributions
        </h1>
        <p className="text-gray-600">
          Explore and visualize various probability distributions.
        </p>
      </div>

      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-2xl font-bold mb-6">Select a Distribution</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {distributions.map((dist) => (
            <div
              key={dist.type}
              onClick={() => handleDistributionChange(dist.type)}
              className={`cursor-pointer p-6 rounded-lg shadow-md transition transform hover:scale-105 ${
                distributionType === dist.type
                  ? "bg-blue-100 border-blue-500 border-2"
                  : "bg-white"
              }`}
            >
              <h3 className="text-lg font-bold">{dist.name}</h3>
              <p className="text-gray-600 mt-2">{dist.description}</p>
            </div>
          ))}
        </div>
      </div>

      {distributionType && (
        <div className="flex flex-col md:flex-row gap-6">
          <div className="md:w-1/3 bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-bold mb-4">Input Parameters</h3>
            <div className="space-y-4">
              {Object.entries(inputs).map(([key, value]) => (
                <div key={key}>
                  <label className="block mb-1 font-medium">
                    {inputLabels[distributionType][key]}
                  </label>
                  <input
                    type="number"
                    className="border p-2 rounded w-full"
                    value={value}
                    onChange={(e) => handleInputChange(key, e.target.value)}
                  />
                </div>
              ))}
              <button
                onClick={simulateDistribution}
                disabled={isLoading}
                className="w-full py-2 mt-4 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
              >
                {isLoading ? <LoadingDots color="white" /> : "Simulate"}
              </button>
            </div>
          </div>

          <div className="md:w-2/3 bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-bold mb-4">Results</h3>
            {isLoading ? (
              <div className="flex justify-center items-center h-64">
                <LoadingDots color="#3498db" />
              </div>
            ) : results ? (
              <>
                <h4 className="text-lg font-semibold mb-4">
                  {results.description}
                </h4>
                <ul className="text-gray-700 space-y-2 mb-6">
                  {Object.entries(results.statistics).map(([key, value]) => (
                    <li key={key}>
                      <strong>{key}:</strong> {value}
                    </li>
                  ))}
                </ul>
                <ProbabilityBarChart
                  data={results.data.frequencies}
                  labels={results.data.labels}
                />
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
};

export default ProbabilityDistributionsPage;
