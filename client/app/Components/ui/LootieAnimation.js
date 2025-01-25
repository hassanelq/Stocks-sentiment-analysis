"use client";

import React from "react";
import { Player } from "@lottiefiles/react-lottie-player";

const LottieAnimation = () => (
  <Player
    autoplay
    loop
    src="/Animations/Animation1.json"
    className="w-full max-w-md mx-auto"
  />
);

export default LottieAnimation;
