import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown, FileText } from 'lucide-react';
import { Button } from '../ui/Button';

export const Hero: React.FC = () => {
  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-ocean-900 via-ocean-800 to-ocean-950 text-white">
      {/* Background decoration */}
      <div className="absolute inset-0 opacity-20">
         <div className="absolute top-0 left-0 w-96 h-96 bg-coral-500 rounded-full blur-[128px] -translate-x-1/2 -translate-y-1/2"></div>
         <div className="absolute bottom-0 right-0 w-[30rem] h-[30rem] bg-ocean-400 rounded-full blur-[128px] translate-x-1/3 translate-y-1/3"></div>
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-4 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <span className="inline-block px-4 py-1.5 mb-6 text-sm font-medium text-ocean-200 bg-ocean-900/50 border border-ocean-700 rounded-full backdrop-blur-sm">
            AI for Marine Conservation
          </span>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6 leading-tight">
            Distilled <span className="text-transparent bg-clip-text bg-gradient-to-r from-coral-400 to-coral-600">Coral Bleaching</span>
          </h1>
          <p className="text-xl md:text-2xl text-ocean-100 mb-10 max-w-2xl mx-auto leading-relaxed">
            Deploying efficient, high-accuracy coral health classification to edge devices using Knowledge Distillation.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button href="/assets/earth_paper.pdf" variant="primary" target="_blank">
              <FileText className="w-5 h-5 mr-2" />
              Read the Paper
            </Button>
            <Button href="#results" variant="outline" className="border-white/20 text-white hover:bg-white/10">
              View Results
            </Button>
          </div>
        </motion.div>
      </div>

      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 text-ocean-300"
      >
        <ArrowDown className="w-6 h-6 animate-bounce" />
      </motion.div>
    </div>
  );
};
