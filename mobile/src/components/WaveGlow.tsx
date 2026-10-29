import React, { useEffect, useRef, useState } from 'react';
import { AccessibilityInfo, Animated, Easing, StyleSheet, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { gradients } from '../theme/tokens';

/**
 * The luminous wave around the composer. Purely decorative: it never implies
 * progress, because it animates only while a request is in flight and the
 * truthful status text is rendered separately. Respects reduced motion.
 */
export function WaveGlow({ active, height = 92 }: { active: boolean; height?: number }) {
  const shift = useRef(new Animated.Value(0)).current;
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    let mounted = true;
    AccessibilityInfo.isReduceMotionEnabled()
      .then((value) => mounted && setReduceMotion(Boolean(value)))
      .catch(() => undefined);
    const sub = AccessibilityInfo.addEventListener('reduceMotionChanged', (value) => setReduceMotion(Boolean(value)));
    return () => {
      mounted = false;
      sub.remove();
    };
  }, []);

  useEffect(() => {
    if (!active || reduceMotion) {
      shift.stopAnimation();
      shift.setValue(0);
      return;
    }
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(shift, { toValue: 1, duration: 2600, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
        Animated.timing(shift, { toValue: 0, duration: 2600, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [active, reduceMotion, shift]);

  const translateX = shift.interpolate({ inputRange: [0, 1], outputRange: [-28, 28] });
  const scaleY = shift.interpolate({ inputRange: [0, 1], outputRange: [0.82, 1.12] });

  return (
    <View pointerEvents="none" style={[styles.wrap, { height }]}>
      <Animated.View style={[styles.wave, { opacity: active ? 0.95 : 0.55, transform: [{ translateX }, { scaleY }] }]}>
        <LinearGradient colors={[...gradients.wave]} start={{ x: 0, y: 0.5 }} end={{ x: 1, y: 0.5 }} style={StyleSheet.absoluteFill} />
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    left: -18,
    right: -18,
    bottom: -18,
    borderRadius: 48,
    overflow: 'hidden',
  },
  wave: { flex: 1, borderRadius: 48 },
});
