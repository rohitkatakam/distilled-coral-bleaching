import React from 'react';
import { Section } from '../ui/Section';

export const Discussion: React.FC = () => {
  return (
    <Section id="discussion" background="light">
      <div className="max-w-3xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-8 text-center">
          Discussion
        </h2>

        <div className="space-y-8">
          <div>
            <h3 className="text-xl font-semibold mb-3">Why Did Student Baseline Outperform Teacher?</h3>
            <p className="text-base text-gray-700 leading-relaxed">
              The student's superior performance (+0.72% despite 15.5× fewer parameters) suggests ResNet50 is overparameterized for binary coral classification. With only 645 training images, the 23.5M-parameter teacher shows a larger generalization gap (5.3%: 83.0% validation → 77.70% test) compared to the student's smaller gap (3.6%: 82.01% → 78.42%), consistent with overfitting. The student's constrained capacity acts as implicit regularization, forcing generalizable feature learning.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Why Did T=2.0, α=0.5 Work Best?</h3>
            <p className="text-base text-gray-700 leading-relaxed">
              Lower temperature (T=2.0) produces sharper distributions that preserve discriminative information for binary classification, while higher temperatures (T=8.0) may obscure the distinction between healthy and bleached corals. This contrasts with multi-class ImageNet defaults where softer distributions reveal inter-class similarities. Balanced weighting (α=0.5) effectively leverages both ground truth and teacher knowledge, while extreme values fail: over-emphasizing hard labels (α=0.3) neglects teacher guidance, while over-relying on the teacher (α=0.7, α=0.9) propagates its weaknesses.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Confidence Calibration</h3>
            <p className="text-base text-gray-700 leading-relaxed">
              The best distilled model (T=2.0, α=0.5) exhibits a confidence gap of 0.153 between correct and incorrect predictions, compared to 0.118 for the student baseline—30% higher. This improvement stems from the teacher's soft probability distributions, which convey uncertainty estimates absent from hard labels. Well-calibrated models enable field deployment where high-confidence predictions proceed autonomously while low-confidence cases trigger expert review.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Limitations</h3>
            <ul className="list-disc list-inside space-y-2 text-base text-gray-700">
              <li>Small dataset (923 images) may limit generalization to other coral species, regions, and imaging conditions</li>
              <li>Single teacher-student architecture pair (ResNet50 → MobileNetV3-Small)</li>
              <li>Strategic sampling of four configurations, not exhaustive hyperparameter search</li>
              <li>No deployment benchmarks on actual edge hardware (latency, energy consumption)</li>
              <li>Single training runs without variance estimates or confidence intervals</li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Future Work</h3>
            <ul className="list-disc list-inside space-y-2 text-base text-gray-700">
              <li>Combine KD with quantization and pruning for greater compression (50-100×)</li>
              <li>Evaluate on actual edge hardware (Raspberry Pi, Jetson, underwater vehicles)</li>
              <li>Cross-domain validation on diverse coral datasets</li>
              <li>Extend to multi-class bleaching severity prediction</li>
              <li>Exhaustive hyperparameter search or Bayesian optimization</li>
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
};
