'use client';

import React, { useState, useEffect, useRef } from 'react';

interface AgentStep {
  step: number;
  state: string;
  thought: string;
  action?: string | null;
  observation?: string | null;
  confidence: string;
}

interface AgentTraceData {
  steps: AgentStep[];
  total_time: number;
  tool_calls: number;
  iterations: number;
  quality_score: number;
}

interface AgentTraceProps {
  trace: AgentTraceData | null;
  isProcessing: boolean;
}

const STATE_ICONS: Record<string, string> = {
  starting: '🚀',
  thinking: '🧠',
  acting: '⚡',
  observing: '👁️',
  reflecting: '🔍',
  refining: '🔬',
  validating: '✅',
  complete: '🎉',
  error: '❌',
};

const STATE_COLORS: Record<string, string> = {
  starting: 'bg-blue-500',
  thinking: 'bg-purple-500',
  acting: 'bg-yellow-500',
  observing: 'bg-green-500',
  reflecting: 'bg-indigo-500',
  refining: 'bg-orange-500',
  validating: 'bg-teal-500',
  complete: 'bg-emerald-500',
  error: 'bg-red-500',
};

const AgentTrace: React.FC<AgentTraceProps> = ({ trace, isProcessing }) => {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new steps arrive
  useEffect(() => {
    if (scrollRef.current && isProcessing) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [trace?.steps?.length, isProcessing]);

  const toggleStep = (stepNum: number) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepNum)) {
      newExpanded.delete(stepNum);
    } else {
      newExpanded.add(stepNum);
    }
    setExpandedSteps(newExpanded);
  };

  if (!trace && !isProcessing) {
    return null;
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="bg-gray-800 px-4 py-3 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
            <span className="font-medium text-white">Agent Reasoning</span>
          </div>
          {trace && (
            <div className="flex items-center gap-4 text-sm text-gray-400">
              <span>🔧 {trace.tool_calls} tools</span>
              <span>⏱️ {trace.total_time.toFixed(1)}s</span>
              <span>📊 {trace.quality_score}%</span>
            </div>
          )}
        </div>
      </div>

      {/* Processing Animation */}
      {isProcessing && (
        <div className="px-4 py-3 bg-gradient-to-r from-blue-900/30 to-purple-900/30 border-b border-gray-700">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <div className="absolute inset-0 flex items-center justify-center text-xs">🤖</div>
            </div>
            <div>
              <div className="text-white font-medium">Agent is thinking...</div>
              <div className="text-gray-400 text-sm">Analyzing document and extracting fields</div>
            </div>
          </div>
        </div>
      )}

      {/* Steps Timeline */}
      <div 
        ref={scrollRef}
        className="max-h-[400px] overflow-y-auto p-4 space-y-3"
      >
        {trace?.steps?.map((step, index) => (
          <div 
            key={step.step}
            className={`
              relative pl-8 pb-3
              ${index < (trace.steps.length - 1) ? 'border-l-2 border-gray-700 ml-2' : 'ml-2'}
            `}
          >
            {/* Step Indicator */}
            <div 
              className={`
                absolute left-0 top-0 -translate-x-1/2 w-5 h-5 rounded-full 
                flex items-center justify-center text-xs
                ${STATE_COLORS[step.state] || 'bg-gray-600'}
              `}
            >
              {STATE_ICONS[step.state] || '•'}
            </div>

            {/* Step Content */}
            <div 
              className={`
                bg-gray-800 rounded-lg p-3 cursor-pointer transition-all
                hover:bg-gray-750 border border-gray-700
                ${expandedSteps.has(step.step) ? 'ring-1 ring-blue-500' : ''}
              `}
              onClick={() => toggleStep(step.step)}
            >
              {/* Step Header */}
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className={`
                      text-xs px-2 py-0.5 rounded-full capitalize
                      ${STATE_COLORS[step.state] || 'bg-gray-600'} text-white
                    `}>
                      {step.state}
                    </span>
                    {step.action && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-blue-600 text-white">
                        {step.action}
                      </span>
                    )}
                  </div>
                  <p className="text-gray-300 text-sm mt-2">{step.thought}</p>
                </div>
                <span className="text-gray-500 text-xs">#{step.step}</span>
              </div>

              {/* Expanded Content */}
              {expandedSteps.has(step.step) && step.observation && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">Observation:</div>
                  <div className="text-gray-400 text-sm font-mono bg-gray-900 p-2 rounded">
                    {step.observation}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Empty State */}
        {!trace?.steps?.length && !isProcessing && (
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-2">🤖</div>
            <p>Upload an image to see the agent in action</p>
          </div>
        )}
      </div>

      {/* Footer Summary */}
      {trace && !isProcessing && (
        <div className="bg-gray-800 px-4 py-3 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-gray-400 text-sm">
              Completed {trace.steps.length} reasoning steps
            </span>
            <div className={`
              px-3 py-1 rounded-full text-sm font-medium
              ${trace.quality_score >= 70 ? 'bg-green-600 text-white' : 
                trace.quality_score >= 40 ? 'bg-yellow-600 text-white' : 
                'bg-red-600 text-white'}
            `}>
              Quality: {trace.quality_score}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentTrace;
