import React from 'react';
import { Section } from '../ui/Section';

export const Methodology: React.FC = () => {
  return (
    <Section id="methodology" background="light">
      <div className="max-w-3xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-8 text-center">
          Methods
        </h2>

        <div className="space-y-8">
          <div>
            <h3 className="text-xl font-semibold mb-3">Dataset</h3>
            <p className="text-base text-gray-700 leading-relaxed">
              Kaggle Coral Bleaching Dataset comprising 923 RGB images labeled as healthy (438 samples, 47.5%) or bleached (485 samples, 52.5%). The dataset was split into training (645 images), validation (139 images), and test (139 images) sets using stratified sampling with random seed 42.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Model Architectures</h3>
            <ul className="space-y-3 text-base text-gray-700">
              <li>
                <strong>Teacher Model:</strong> ResNet50 pretrained on ImageNet (23.5M parameters), modified for binary classification with dropout (p=0.5).
              </li>
              <li>
                <strong>Student Model:</strong> MobileNetV3-Small (1.52M parameters, 15.5× reduction), modified for binary classification.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Training Procedure</h3>
            <p className="text-base text-gray-700 leading-relaxed mb-3">
              I manually implemented the distillation mechanism with temperature-scaled softmax and KL divergence between teacher and student soft targets. The combined loss function blends hard-label cross-entropy with soft-target KL divergence:
            </p>
            <div className="bg-gray-50 p-4 border border-gray-300 font-mono text-sm">
              L<sub>KD</sub> = α · L<sub>hard</sub> + (1 − α) · T² · L<sub>soft</sub>
            </div>
            <p className="text-base text-gray-700 leading-relaxed mt-3">
              All models were trained using Adam optimizer (learning rate 1×10⁻⁴) with early stopping (patience 10 epochs) on Google Colab T4 GPUs.
            </p>
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-3">Hyperparameter Exploration</h3>
            <p className="text-base text-gray-700 leading-relaxed">
              I explored temperature <em>T</em> ∈ {'{2.0, 4.0, 8.0}'} and weighting factor <em>α</em> ∈ {'{0.3, 0.5, 0.7, 0.9}'} using four strategic configurations to identify optimal distillation settings for binary coral classification.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
};
