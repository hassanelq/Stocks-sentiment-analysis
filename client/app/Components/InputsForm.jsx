"use client";

import React from "react";

const InputsForm = ({ inputs, setInputs, type }) => {
  const fieldLabels = {
    investment: {
      initialInvestment: "Initial Investment",
      annualReturn: "Annual Return (%)",
      annualVolatility: "Annual Volatility (%)",
      timeHorizon: "Time Horizon (years)",
      annualContribution: "Annual Contribution",
      investmentGoal: "Target Investment Value",
      numSimulations: "Number of Simulations",
    },
    options: {
      stockPrice: "Stock Price",
      strikePrice: "Strike Price",
      riskFreeRate: "Risk-Free Rate (%)",
      volatility: "Volatility (%)",
      timeToMaturity: "Time to Maturity (years)",
      numSimulations: "Number of Simulations",
    },
    risk: {
      portfolioValue: "Portfolio Value",
      expectedReturn: "Expected Return (%)",
      volatility: "Volatility (%)",
      timeHorizon: "Time Horizon (years)",
      numSimulations: "Number of Simulations",
    },
  };

  if (!type || !fieldLabels[type]) {
    console.error("Invalid simulation type provided to InputsForm");
    return <p>Invalid simulation type selected.</p>;
  }

  return (
    <div>
      {Object.entries(fieldLabels[type]).map(([key, label]) => (
        <div key={key} className="mb-4">
          <label className="block font-medium">{label}</label>
          <input
            type="number"
            value={inputs[key] || 0} // Default to 0
            onChange={(e) =>
              setInputs((prev) => ({
                ...prev,
                [key]: parseFloat(e.target.value) || 0, // Ensure numeric value
              }))
            }
            className="border p-2 rounded w-full"
          />
        </div>
      ))}
    </div>
  );
};

export default InputsForm;
