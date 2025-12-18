import React from 'react';
import { Section } from '../ui/Section';

export const Abstract: React.FC = () => {
  return (
    <Section id="abstract" background="light">
      <div className="max-w-3xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-8 text-center">
          Introduction
        </h2>

        <div className="space-y-6 text-base md:text-lg leading-relaxed text-gray-700">
          <p>
            Coral bleaching, a stress response where corals expel their symbiotic algae, threatens reef ecosystems worldwide and serves as an indicator of ocean health. Automated image classification using computer vision enables easier coral reef monitoring, but deploying deep learning models onto edge devices in resource-constrained field environments remains challenging.
          </p>

          <p>
            Knowledge distillation addresses this challenge by transferring knowledge from a large "teacher" model to a compact "student" model through soft probability distributions produced by temperature scaling. This technique enables student models to mimic a teacher's behavior on specific tasks such as binary image classification while being much less computationally intensive.
          </p>

          <p>
            In this project, I compressed a ResNet50 teacher model (23.5M parameters) into a MobileNetV3-Small student (1.52M parameters)—a 15.5× reduction—using the Kaggle Coral Bleaching Dataset (923 images). I systematically explored the temperature <em>T</em> and weighting parameter <em>α</em> hyperparameter space to identify optimal distillation configurations.
          </p>

          <div className="mt-8 p-6 border border-gray-300 bg-gray-50">
            <p className="text-lg font-semibold text-center text-black">
              Key Finding: 79.14% test accuracy with 15.5× parameter compression
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
};
