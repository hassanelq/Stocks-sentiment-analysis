import Hero from "./Components/HeroSection";
import CTA from "./Components/CTA";
import GradientWrapper from "./Components/GradientWrapper";
import Contributors from "./Components/devs";
export default function Home() {
  return (
    <>
      <Hero />
      <GradientWrapper></GradientWrapper>
      <CTA />
      <GradientWrapper></GradientWrapper>
      <Contributors />
    </>
  );
}
