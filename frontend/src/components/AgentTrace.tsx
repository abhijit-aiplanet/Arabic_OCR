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

const STATE_LABELS: Record<string, string> = {
  starting: 'Initializing',
  thinking: 'Reasoning',
  acting: 'Executing',
  observing: 'Processing',
  reflecting: 'Analyzing',
  refining: 'Refining',
  validating: 'Validating',
  complete: 'Complete',
  error: 'Error',
};

const AgentTrace: React.FC<AgentTraceProps> = ({ trace, isProcessing }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current && isProcessing) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [trace?.steps?.length, isProcessing]);

  if (!trace && !isProcessing) {
    return null;
  }

  const qualityColor = trace?.quality_score 
    ? trace.quality_score >= 70 ? 'text-emerald-600' 
    : trace.quality_score >= 40 ? 'text-amber-600' 
    : 'text-red-600'
    : 'text-gray-400';

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm">
      {/* Header */}
      <div 
        className="px-5 py-4 border-b border-gray-100 cursor-pointer hover:bg-gray-50/50 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isProcessing ? (
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
            ) : (
              <div className={`w-2 h-2 rounded-full ${trace?.quality_score && trace.quality_score >= 50 ? 'bg-emerald-500' : 'bg-amber-500'}`} />
            )}
            <span className="text-sm font-medium text-gray-900">Agent Execution</span>
          </div>
          
          <div className="flex items-center gap-6">
            {trace && (
              <>
                <div className="flex items-center gap-1.5 text-xs text-gray-500">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>{trace.tool_calls}</span>
                </div>
                <div className="flex items-center gap-1.5 text-xs text-gray-500">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{trace.total_time.toFixed(1)}s</span>
                </div>
                <div className={`text-xs font-medium ${qualityColor}`}>
                  {trace.quality_score}%
                </div>
              </>
            )}
            <svg 
              className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>

      {/* Processing State */}
      {isProcessing && (
        <div className="px-5 py-3 bg-gray-50 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="relative w-4 h-4">
              <div className="absolute inset-0 border-2 border-gray-200 rounded-full" />
              <div className="absolute inset-0 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            </div>
            <span className="text-sm text-gray-600">Processing document...</span>
          </div>
        </div>
      )}

      {/* Steps */}
      {isExpanded && (
        <div 
          ref={scrollRef}
          className="max-h-[320px] overflow-y-auto"
        >
          {trace?.steps?.length ? (
            <div className="divide-y divide-gray-50">
              {trace.steps.map((step, index) => (
                <div 
                  key={step.step}
                  className={`px-5 py-3 cursor-pointer transition-colors ${
                    selectedStep === step.step ? 'bg-gray-50' : 'hover:bg-gray-50/50'
                  }`}
                  onClick={() => setSelectedStep(selectedStep === step.step ? null : step.step)}
                >
                  <div className="flex items-start gap-3">
                    {/* Step Number */}
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center">
                      <span className="text-xs font-medium text-gray-500">{step.step}</span>
                    </div>
                    
                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                          step.state === 'complete' ? 'bg-emerald-50 text-emerald-700' :
                          step.state === 'error' ? 'bg-red-50 text-red-700' :
                          step.state === 'acting' ? 'bg-blue-50 text-blue-700' :
                          'bg-gray-100 text-gray-600'
                        }`}>
                          {STATE_LABELS[step.state] || step.state}
                        </span>
                        {step.action && (
                          <span className="text-xs text-gray-400 font-mono">{step.action}</span>
                        )}
                      </div>
                      <p className="text-sm text-gray-700 line-clamp-2">{step.thought}</p>
                      
                      {/* Expanded Details */}
                      {selectedStep === step.step && step.observation && (
                        <div className="mt-3 p-3 bg-gray-100 rounded-lg">
                          <div className="text-xs font-medium text-gray-500 mb-1">Output</div>
                          <p className="text-xs text-gray-600 font-mono whitespace-pre-wrap break-all">
                            {step.observation}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : !isProcessing ? (
            <div className="px-5 py-8 text-center">
              <div className="text-gray-400 text-sm">No execution data</div>
            </div>
          ) : null}
        </div>
      )}

      {/* Footer */}
      {trace && !isProcessing && isExpanded && (
        <div className="px-5 py-3 bg-gray-50 border-t border-gray-100">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>{trace.steps.length} steps completed</span>
            <span className={`font-medium ${qualityColor}`}>
              Quality Score: {trace.quality_score}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentTrace;
