"use client";
import NavLink from "./ui/NavLink";
import dynamic from "next/dynamic";

// Dynamically import LottieAnimation
const LottieAnimation = dynamic(() => import("./ui/LootieAnimation"), {
  ssr: false, // Disable SSR
});

export default function Hero() {
  return (
    <section>
      <div className="custom-screen text-gray-600 text-center flex flex-col gap-14 pt-14">
        <div className="space-y-8 max-w-4xl mx-auto">
          {/* Lottie Animation */}
          <div className="flex justify-center">
            <LottieAnimation />
          </div>

          {/* Hero Content */}
          <div className="space-y-5">
            <h1 className="text-4xl text-gray-800 font-extrabold mx-auto sm:text-6xl">
              Welcome to Our Financial Simulation Platform
            </h1>
            <p className="max-w-xl mx-auto">
              Discover a powerful and interactive tool designed to help you
              explore quantitative finance concepts. Simulate financial models,
              including Normal Distributions, Brownian Motion, and Monte Carlo
              simulations, to better understand market dynamics and asset
              behavior.
            </p>
            <div className="flex items-center justify-center gap-x-3 font-medium text-sm">
              <NavLink
                href="/start"
                className="text-white bg-gray-800 hover:bg-gray-600 active:bg-gray-900"
              >
                Start Simulations
              </NavLink>
              <NavLink
                target="_blank"
                href="https://github.com/hassanelq/Simulations-MC-BrownMotion"
                className="text-gray-700 border hover:bg-gray-50"
                scroll={false}
              >
                Learn More
              </NavLink>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
