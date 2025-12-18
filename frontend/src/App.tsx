import React from 'react';
import { Hero } from './components/sections/Hero';
import { Abstract } from './components/sections/Abstract';
import { Methodology } from './components/sections/Methodology';
import { Results } from './components/sections/Results';
import { Footer } from './components/sections/Footer';

function App() {
  return (
    <div className="min-h-screen bg-ocean-50 overflow-x-hidden font-sans">
      <Hero />
      <Abstract />
      <Methodology />
      <Results />
      <Footer />
    </div>
  );
}

export default App;
