import React from 'react';
import { Section } from '../ui/Section';
import { FigureViewer } from '../ui/FigureViewer';
import { motion } from 'framer-motion';

export const Results: React.FC = () => {
  const stats = [
    { label: "Student Accuracy", value: "79.14%", sub: "+1.44% vs Teacher", color: "text-teal-500" },
    { label: "Model Compression", value: "15.5x", sub: "Smaller Footprint", color: "text-coral-500" },
    { label: "Confidence Gap", value: "0.153", sub: "Better Calibration", color: "text-ocean-500" }
  ];

  return (
    <Section id="results" background="light">
      <div className="text-center mb-12">
        <h2 className="text-3xl md:text-4xl font-bold text-ocean-900 mb-4">Results & Analysis</h2>
        <p className="text-ocean-600 max-w-2xl mx-auto">
          Our distilled Student model not only matches but exceeds the Teacher's accuracy while being significantly more efficient.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
        {stats.map((stat, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="bg-white rounded-2xl p-8 shadow-xl text-center border border-ocean-50"
          >
            <div className={`text-4xl md:text-5xl font-black mb-2 ${stat.color}`}>{stat.value}</div>
            <div className="text-lg font-bold text-ocean-800 mb-1">{stat.label}</div>
            <div className="text-sm text-ocean-500 font-medium">{stat.sub}</div>
          </motion.div>
        ))}
      </div>

      {/* Main Figure Showcase */}
      <div className="mb-16">
        <h3 className="text-2xl font-bold text-ocean-900 mb-6 border-l-4 border-ocean-500 pl-4">Performance Overview</h3>
        <FigureViewer 
          src="/assets/images/main_results.png" 
          alt="Main Results Comparison" 
          caption="Panel A: Accuracy Comparison | Panel B: Parameter Count | Panel C: Performance vs Efficiency | Panel D: KD Effectiveness"
          className="rounded-2xl shadow-2xl border-4 border-white"
        />
      </div>

      {/* Secondary Figures Grid */}
      <div className="grid md:grid-cols-2 gap-8">
        <div>
          <h3 className="text-xl font-bold text-ocean-900 mb-4 border-l-4 border-teal-500 pl-4">Confusion Matrices</h3>
          <FigureViewer 
            src="/assets/images/confusion_matrices_comparison.png" 
            alt="Confusion Matrices" 
            caption="Comparison of Teacher, Baseline Student, and Distilled Student confusion matrices."
          />
        </div>
        <div>
          <h3 className="text-xl font-bold text-ocean-900 mb-4 border-l-4 border-coral-500 pl-4">Calibration Analysis</h3>
          <FigureViewer 
            src="/assets/images/confidence_by_correctness.png" 
            alt="Confidence Analysis" 
            caption="The Distilled Student (T=2.0) shows the highest confidence gap between correct and incorrect predictions, indicating better reliability."
          />
        </div>
      </div>
    </Section>
  );
};
