import { Hero } from './components/sections/Hero';
import { Abstract } from './components/sections/Abstract';
import { Methodology } from './components/sections/Methodology';
import { Results } from './components/sections/Results';
import { Discussion } from './components/sections/Discussion';
import { Footer } from './components/sections/Footer';

function App() {
  return (
    <div className="min-h-screen bg-white overflow-x-hidden font-sans">
      <Hero />
      <Abstract />
      <Methodology />
      <Results />
      <Discussion />
      <Footer />
    </div>
  );
}

export default App;
