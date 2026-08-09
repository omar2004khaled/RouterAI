import type { Action, MessageType } from './routing';
export interface Message { id: string; sender: string; conversation: string; text: string; type: MessageType; action: Action; confidence: number; timestamp: string; source?: 'dataset' | 'gmail'; }
export interface DashboardStats { total: number; notify: number; digest: number; mute: number; confidence: number; processingTime: number | null; }
