import React, { useEffect, useRef } from 'react';
import { BackHandler, Platform, StyleSheet, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { ChatScreen } from './src/screens/ChatScreen';

export default function App() {
  const lastBack = useRef(0);

  useEffect(() => {
    if (Platform.OS !== 'android') return;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      // The drawer and settings sheet handle their own dismissal through Modal.
      // Here we only make sure a second press exits, so navigation is predictable.
      const now = Date.now();
      if (now - lastBack.current < 2000) return false;
      lastBack.current = now;
      return true;
    });
    return () => sub.remove();
  }, []);

  return (
    <SafeAreaProvider>
      <View style={styles.root}>
        <StatusBar style="auto" />
        <ChatScreen />
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
});
