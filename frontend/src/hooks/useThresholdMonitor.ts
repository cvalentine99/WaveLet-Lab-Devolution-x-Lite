import { useEffect, useRef } from 'react';
import { useThresholdStore } from '@/stores/thresholdStore';
import { useViolationLogStore } from '@/stores/violationLogStore';
import { toast } from 'sonner';

// Singleton AudioContext to prevent memory leaks (browser limits ~6 concurrent contexts)
let sharedAudioContext: AudioContext | null = null;

const getAudioContext = (): AudioContext | null => {
  if (!sharedAudioContext) {
    try {
      sharedAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    } catch (error) {
      console.error('Failed to create AudioContext:', error);
      return null;
    }
  }
  
  // Resume if suspended (browser autoplay policy)
  if (sharedAudioContext.state === 'suspended') {
    sharedAudioContext.resume().catch(console.error);
  }
  
  return sharedAudioContext;
};

// Simple beep sound using Web Audio API
const playAlertSound = () => {
  try {
    const audioContext = getAudioContext();
    if (!audioContext) return;
    
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.value = 800; // 800 Hz tone
    oscillator.type = 'sine';

    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
    
    // Clean up oscillator after it stops (gainNode persists with context)
    oscillator.onended = () => oscillator.disconnect();
  } catch (error) {
    console.error('Failed to play alert sound:', error);
  }
};

interface Detection {
  center_freq_hz: number;
  power_dbm: number;
  snr_db: number;
}

export function useThresholdMonitor(detections: Detection[]) {
  const { getEnabledThresholds, incrementTrigger } = useThresholdStore();
  const { addViolation } = useViolationLogStore();
  const lastAlertTime = useRef<Map<string, number>>(new Map());
  const ALERT_COOLDOWN_MS = 5000; // 5 seconds between alerts for same threshold

  useEffect(() => {
    const thresholds = getEnabledThresholds();
    if (thresholds.length === 0 || detections.length === 0) return;

    const now = Date.now();

    detections.forEach((detection) => {
      thresholds.forEach((threshold) => {
        // Check if detection is within frequency range
        const detectionFreq = detection.center_freq_hz;
        if (detectionFreq < threshold.freqStartHz || detectionFreq > threshold.freqEndHz) {
          return;
        }

        // Check if power exceeds threshold
        const detectionPower = detection.power_dbm || detection.snr_db;
        if (detectionPower < threshold.thresholdDbm) {
          return;
        }

        // Check cooldown
        const lastAlert = lastAlertTime.current.get(threshold.id) || 0;
        if (now - lastAlert < ALERT_COOLDOWN_MS) {
          return;
        }

        // Trigger alert!
        lastAlertTime.current.set(threshold.id, now);
        incrementTrigger(threshold.id);

        // Log violation
        addViolation({
          thresholdId: threshold.id,
          thresholdName: threshold.name,
          centerFreqHz: detectionFreq,
          powerDbm: detectionPower,
          thresholdDbm: threshold.thresholdDbm,
          exceedanceDbm: detectionPower - threshold.thresholdDbm,
        });

        const freqMHz = (detectionFreq / 1e6).toFixed(2);
        const message = `${threshold.name}: ${detectionPower.toFixed(1)} dBm at ${freqMHz} MHz`;

        if (threshold.alertType === 'toast' || threshold.alertType === 'both') {
          toast.warning(message, {
            duration: 4000,
            description: `Threshold: ${threshold.thresholdDbm} dBm`,
          });
        }

        if (threshold.alertType === 'audio' || threshold.alertType === 'both') {
          playAlertSound();
        }
      });
    });
  }, [detections, getEnabledThresholds, incrementTrigger, addViolation]);
}
