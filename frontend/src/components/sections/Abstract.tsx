import React from 'react';
import { Card } from '../ui/Card';
import { Section } from '../ui/Section';
import { AlertTriangle, Server, Zap } from 'lucide-react';

export const Abstract: React.FC = () => {
  return (
    <Section id="abstract" background="light">
      <div className="grid md:grid-cols-2 gap-12 items-center">
        <div>
          <h2 className="text-3xl md:text-4xl font-bold text-ocean-900 mb-6">
            The Challenge
          </h2>
          <p className="text-lg text-ocean-700 mb-6 leading-relaxed">
            Coral reefs are dying at alarming rates due to climate change. Monitoring their health requires analyzing millions of images, but traditional deep learning models are too heavy for the autonomous underwater vehicles (AUVs) and edge buoys used in the field.
          </p>
          <p className="text-lg text-ocean-700 mb-6 leading-relaxed">
            We propose a <strong>Knowledge Distillation</strong> pipeline that compresses a massive ResNet50 "Teacher" into a lightweight MobileNetV3 "Student", retaining <strong>99%</strong> of the accuracy with <strong>15.5x</strong> fewer parameters.
          </p>
        </div>
        
        <div className="grid gap-6">
          <Card delay={0.2} className="border-l-4 border-l-coral-500">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-coral-50 rounded-lg text-coral-600">
                <AlertTriangle size={24} />
              </div>
              <div>
                <h3 className="text-lg font-bold text-ocean-900 mb-1">Ecological Crisis</h3>
                <p className="text-ocean-600">Mass bleaching events require real-time monitoring to direct conservation efforts effectively.</p>
              </div>
            </div>
          </Card>

          <Card delay={0.3} className="border-l-4 border-l-ocean-500">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-ocean-50 rounded-lg text-ocean-600">
                <Server size={24} />
              </div>
              <div>
                <h3 className="text-lg font-bold text-ocean-900 mb-1">Resource Constraints</h3>
                <p className="text-ocean-600">Edge devices have limited battery and compute power. Large models drain batteries too fast.</p>
              </div>
            </div>
          </Card>

          <Card delay={0.4} className="border-l-4 border-l-teal-500">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-teal-50 rounded-lg text-teal-600">
                <Zap size={24} />
              </div>
              <div>
                <h3 className="text-lg font-bold text-ocean-900 mb-1">Our Solution</h3>
                <p className="text-ocean-600">Distilled Student models that are fast, small, and highly accurate.</p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
};
