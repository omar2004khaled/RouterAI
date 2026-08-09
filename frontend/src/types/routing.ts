export type Action = 'NOTIFY' | 'DIGEST' | 'MUTE';
export type MessageType = 'text' | 'image' | 'voice' | 'document';
export interface Evidence { id: string; text: string; signal: string; }
export interface AnalysisResult { action: Action; confidence: number; message_type: MessageType; priority: 'Critical' | 'High' | 'Normal' | 'Low'; reasoning: string; evidence: Evidence[]; rules_triggered: string[]; processing_time: number; }
