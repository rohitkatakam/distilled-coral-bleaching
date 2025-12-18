import React from 'react';
import { Section } from '../ui/Section';
import { motion } from 'framer-motion';

export const Methodology: React.FC = () => {
  return (
    <Section id="methodology" background="dark">
      <div className="text-center mb-16">
        <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">Methodology</h2>
        <p className="text-ocean-200 max-w-2xl mx-auto text-lg">
          Knowledge Distillation transfers the generalization capability of a large, complex Teacher model to a small, efficient Student model using soft targets.
        </p>
      </div>

      <div className="relative max-w-4xl mx-auto py-10">
        {/* Diagram Container */}
        <div className="bg-ocean-800/50 rounded-2xl p-8 md:p-12 border border-ocean-700 backdrop-blur-sm">
            <div className="flex flex-col md:flex-row items-center justify-between gap-12 relative z-10">
                
                {/* Teacher Node */}
                <div className="text-center w-full md:w-1/3">
                    <motion.div 
                        initial={{ scale: 0.8, opacity: 0 }}
                        whileInView={{ scale: 1, opacity: 1 }}
                        transition={{ delay: 0.2 }}
                        className="bg-coral-500/10 border-2 border-coral-500 rounded-xl p-6 mb-4 relative"
                    >
                        <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-coral-500 text-white text-xs font-bold px-3 py-1 rounded-full">
                            TEACHER
                        </div>
                        <div className="text-4xl font-black text-coral-400 mb-2">23.5M</div>
                        <div className="text-sm text-ocean-200">Parameters</div>
                        <div className="font-mono text-xs text-ocean-400 mt-2">ResNet50</div>
                    </motion.div>
                </div>

                {/* Arrow / Process */}
                <div className="flex flex-col items-center justify-center text-center w-full md:w-1/3">
                    <div className="text-ocean-300 text-sm font-medium mb-2">KL Divergence Loss</div>
                    <motion.div 
                        initial={{ width: 0 }}
                        whileInView={{ width: "100%" }}
                        transition={{ delay: 0.5, duration: 0.8 }}
                        className="h-1 bg-gradient-to-r from-coral-500 to-teal-500 rounded-full w-full mb-2"
                    />
                    <div className="text-ocean-300 text-xs">Temperature (T=2.0)</div>
                </div>

                {/* Student Node */}
                <div className="text-center w-full md:w-1/3">
                    <motion.div 
                        initial={{ scale: 0.8, opacity: 0 }}
                        whileInView={{ scale: 1, opacity: 1 }}
                        transition={{ delay: 0.8 }}
                        className="bg-teal-500/10 border-2 border-teal-500 rounded-xl p-6 mb-4 relative"
                    >
                        <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-teal-500 text-white text-xs font-bold px-3 py-1 rounded-full">
                            STUDENT
                        </div>
                        <div className="text-4xl font-black text-teal-400 mb-2">1.52M</div>
                        <div className="text-sm text-ocean-200">Parameters</div>
                        <div className="font-mono text-xs text-ocean-400 mt-2">MobileNetV3</div>
                    </motion.div>
                </div>
            </div>
        </div>
      </div>
    </Section>
  );
};
