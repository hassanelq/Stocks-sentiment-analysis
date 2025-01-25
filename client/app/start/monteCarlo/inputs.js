import React from "react";

const InputsForm = ({ inputs, setInputs }) => {
  const fields = {
    investment: [
      "initialInvestment",
      "annualReturn",
      "annualVolatility",
      "timeHorizon",
      "annualContribution",
    ],
    options: [
      "stockPrice",
      "strikePrice",
      "riskFreeRate",
      "volatility",
      "timeToMaturity",
    ],
    risk: ["portfolioValue", "expectedReturn", "volatility", "timeHorizon"],
  };

  return (
    <div>
      {Object.keys(fields).map((key) => (
        <div key={key}>
          {fields[key].map((field) => (
            <div key={field} className="mb-4">
              <label className="block font-medium">{field}</label>
              <input
                type="number"
                value={inputs[field] || ""}
                onChange={(e) =>
                  setInputs((prev) => ({
                    ...prev,
                    [field]: parseFloat(e.target.value),
                  }))
                }
                className="border p-2 rounded w-full"
              />
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

export default InputsForm;
