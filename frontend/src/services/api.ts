import type { AnalysisResult, MessageType } from '../types/routing';
import type { DashboardStats, Message } from '../types/message';

// Point VITE_API_URL at a deployed FastAPI host when moving beyond local development.
const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  if (!response.ok) throw new Error(`Router API request failed (${response.status})`);
  return response.json() as Promise<T>;
}

export const getMessages = () => request<Message[]>('/messages');
export const getDashboardStats = () => request<DashboardStats>('/dashboard');
export const getGmailDashboard = () => request<GmailDashboard>('/dashboard/gmail');
export const getAnalytics = () => request<{ routing: {name: string; value: number; color: string}[]; trend: {day: string; notify: number; digest: number; mute: number}[] }>('/analytics');
export const getSystemStatus = () => request<{status: string; version: string; backend: string; latency: string}>('/system');

export interface GmailDashboard { connected: boolean; email?: string; total: number; notify: number; digest: number; mute: number; confidence: number; analyzed: boolean; }

export function analyzeMessage(input: { text: string; sender: string; type: MessageType }): Promise<AnalysisResult> {
  return request<AnalysisResult>('/analyze', {
    method: 'POST',
    body: JSON.stringify({ text: input.text, sender: input.sender, message_type: input.type, has_media: input.type !== 'text' }),
  });
}

export interface GmailStatus { connected: boolean; email?: string }
export interface GmailResult extends AnalysisResult { id: string; sender: string; subject: string; message: string; preview: string; timestamp: string; source: 'gmail' }
export const getGmailStatus = () => request<GmailStatus>('/gmail/status');
export const connectGmail = () => { window.location.assign(`${API_URL}/auth/gmail`); };
export const disconnectGmail = () => request<{message: string}>('/gmail/connection', { method: 'DELETE' });
export const analyzeGmail = (limit: number) => request<{total: number; results: GmailResult[]}>('/gmail/analyze?limit=' + limit, { method: 'POST' });
export const getGmailResults = () => request<{results: GmailResult[]}>('/gmail/results');
export const submitFeedback = (body: {message_id: string; original_action: string; correct_action: string; subject?: string; sender?: string}) =>
  request<{saved: boolean}>('/gmail/feedback', { method: 'POST', body: JSON.stringify(body) });
