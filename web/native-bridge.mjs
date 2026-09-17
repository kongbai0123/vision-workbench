// The sole browser/desktop boundary. Legacy names are compatibility aliases.
export const nativeCallbacks = Object.create(null);
const names = {
  workbenchUpdateStatus: 'updateStatus', workbenchUpdateProgress: 'updateProgress',
  workbenchFlush: 'flush', workbenchState: 'state', workbenchNativeDrag: 'drag',
  workbenchExternalSync: 'externalSync', workbenchExternalNavigate: 'navigate',
};

export function installNativeCallbacks(target = globalThis) {
  for (const [name, callback] of Object.entries(names)) {
    target[name] = (...args) => nativeCallbacks[callback]?.(...args);
  }
}

export async function connectNative(target = globalThis) {
  if (!target.qt?.webChannelTransport || !target.QWebChannel) return null;
  return new Promise(resolve => new target.QWebChannel(target.qt.webChannelTransport,
    channel => resolve(channel.objects.workbenchNative)));
}
