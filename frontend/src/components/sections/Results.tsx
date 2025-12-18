import React from 'react';
import { Section } from '../ui/Section';
import { FigureViewer } from '../ui/FigureViewer';

export const Results: React.FC = () => {
  return (
    <Section id="results" background="light">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-8 text-center">
          Results
        </h2>

        <div className="space-y-8">
          {/* Key Finding */}
          <div>
            <h3 className="text-xl font-semibold mb-3">Overall Performance</h3>
            <p className="text-base text-gray-700 leading-relaxed mb-4">
              The best knowledge distillation configuration (T=2.0, α=0.5) achieved 79.14% test accuracy with 79.75% precision and 79.48% recall, demonstrating balanced per-class performance.
            </p>

            {/* Comparison Table */}
            <div className="overflow-x-auto border border-gray-300">
              <table className="w-full text-sm">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="px-4 py-3 text-left font-semibold">Model</th>
                    <th className="px-4 py-3 text-right font-semibold">Test Accuracy</th>
                    <th className="px-4 py-3 text-right font-semibold">Parameters</th>
                    <th className="px-4 py-3 text-right font-semibold">Checkpoint Size</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  <tr className="hover:bg-gray-50">
                    <td className="px-4 py-3">Teacher (ResNet50)</td>
                    <td className="px-4 py-3 text-right">77.70%</td>
                    <td className="px-4 py-3 text-right">23.5M</td>
                    <td className="px-4 py-3 text-right">270 MB</td>
                  </tr>
                  <tr className="hover:bg-gray-50">
                    <td className="px-4 py-3">Student Baseline</td>
                    <td className="px-4 py-3 text-right">78.42%</td>
                    <td className="px-4 py-3 text-right">1.52M</td>
                    <td className="px-4 py-3 text-right">18 MB</td>
                  </tr>
                  <tr className="hover:bg-gray-50 bg-gray-50 font-semibold">
                    <td className="px-4 py-3">Best KD Student (T=2.0, α=0.5)</td>
                    <td className="px-4 py-3 text-right">79.14%</td>
                    <td className="px-4 py-3 text-right">1.52M</td>
                    <td className="px-4 py-3 text-right">18 MB</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-base text-gray-700 leading-relaxed mt-4">
              The distilled student achieved <strong>+0.72% improvement over the baseline student</strong> and <strong>+1.44% improvement over the teacher</strong> while maintaining a <strong>15.5× parameter reduction</strong> and <strong>15× smaller checkpoint size</strong>.
            </p>
          </div>

          {/* Main Figure */}
          <div>
            <h3 className="text-xl font-semibold mb-4">Performance Overview</h3>
            <FigureViewer
              src="/figures/main_results.png"
              alt="Main Results: Knowledge Distillation for Coral Bleaching Classification"
              caption="Figure 2: Comprehensive results summary. (A) Test accuracy comparison across all models. (B) Model size comparison showing 15.5× compression. (C) Accuracy vs efficiency tradeoff. (D) KD effectiveness showing improvement over baseline for different hyperparameter configurations."
              className="border border-gray-300"
            />
          </div>

          {/* Hyperparameter Sensitivity */}
          <div>
            <h3 className="text-xl font-semibold mb-3">Hyperparameter Sensitivity</h3>
            <p className="text-base text-gray-700 leading-relaxed">
              Lower temperature (T=2.0) produced sharper distributions that preserved discriminative information for binary classification, outperforming moderate (T=4.0) and high (T=8.0) temperatures. Balanced weighting (α=0.5) effectively leveraged both ground truth and teacher knowledge, while the standard literature default (T=4.0, α=0.7) underperformed at 76.26% accuracy.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
};
