import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown, FileText } from 'lucide-react';
import { Button } from '../ui/Button';

export const Hero: React.FC = () => {
  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden bg-white text-black">
      <div className="relative z-10 max-w-5xl mx-auto px-4 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <span className="inline-block px-4 py-1.5 mb-8 text-xs font-medium text-gray-600 border border-gray-300 rounded-full uppercase tracking-wider">
            AI for Marine Conservation
          </span>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6 leading-tight">
            Distilled Coral Bleaching
          </h1>
          <p className="text-lg md:text-xl text-gray-700 mb-4 max-w-3xl mx-auto font-medium">
            Knowledge Distillation for Coral Bleaching Classification
          </p>
          <p className="text-base md:text-lg text-gray-600 mb-12 max-w-3xl mx-auto leading-relaxed">
            Compressing ResNet50 into MobileNetV3-Small with 15.5× fewer parameters while improving accuracy through knowledge distillation
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button href="/earth_paper.pdf" variant="primary" target="_blank" rel="noopener noreferrer">
              <FileText className="w-5 h-5 mr-2" />
              Read Full Paper
            </Button>
            <Button href="#results" variant="outline">
              View Results
            </Button>
          </div>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 text-gray-400"
      >
        <ArrowDown className="w-6 h-6 animate-bounce" />
      </motion.div>
    </div>
  );
};
