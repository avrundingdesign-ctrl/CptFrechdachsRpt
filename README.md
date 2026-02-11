Dieses Update führt das sofort laden der Yolo Modelle ein. 
Außerdem Yolo26 Modell zur Darts erkennung 

Probleme laut KI: 

🔴 Kritische Bugs
1. Erster Dart jeder neuen Runde geht verloren
Datei: CameraModel.swift

countBefore wird vor dem merge() gespeichert (z.B. 3 von der letzten Runde). Nach dem Reset im Tracker ist currentDarts.count = 1, aber 1 > 3 ist false → die Notification wird nie gefeuert. Der Score des ersten Darts jeder neuen Runde geht komplett verloren.

Fix: countBefore muss den internen Reset des Trackers berücksichtigen, oder einen separaten „last notified count" verwenden.

2. Mid-Turn-Bust korrumpiert den nächsten Spieler
Datei: ContentView.swift

Bei einem Bust durch Dart 1 oder 2 wird sofort nextPlayer() aufgerufen. Wenn dann Dart 2/3 per .Throw-Notification eintrifft (via DispatchQueue.main.async), wird der Score dem falschen Spieler abgezogen.

Fix: Ein turnBusted-Flag einführen, das thrown() und handleTurnFinished() dazu bringt, verbleibende Darts eines gebusteten Turns zu ignorieren.

3. isThrowBusted bleibt nach Mid-Turn-Bust auf true
Datei: ContentView.swift

Im Bust-Fall von thrown() fehlt cameraModel.isThrowBusted = false nach nextPlayer(). In handleTurnFinished() wird es korrekt zurückgesetzt — hier aber nicht. Der DartTracker verhält sich danach falsch.

4. NotificationCenter-Observer sammeln sich an (Duplikate!)
Datei: ContentView.swift

In .onAppear werden Observer hinzugefügt, aber nie entfernt. Jedes Mal, wenn die View erscheint, kommen neue dazu. Ergebnis: handleTurnFinished() und thrown() werden N-mal aufgerufen → Score-Abzüge multiplizieren sich.

Fix: Observer-Tokens speichern und in .onDisappear entfernen, oder einen setupOnce-Guard verwenden.

🟠 Schwere Probleme
5. AVCapturePhotoCaptureDelegate auf Background-Thread in @MainActor-Klasse
Datei: CameraModel.swift

photoOutput(_:didFinishProcessingPhoto:) wird von AVFoundation auf einem Background-Thread aufgerufen. Dort wird aber isCapturingNow (eine @MainActor-Property) modifiziert → Data Race.

Fix: Body in DispatchQueue.main.async { ... } wrappen.

6. AVSpeechSynthesizerDelegate auf falschem Thread
Datei: CameraModel.swift

speechSynthesizer(_:didStart:) setzt isSpeaking = true, aber diese Delegate-Methode läuft nicht zwingend auf dem Main-Thread. Eine @Published-Property von einem Background-Thread zu setzen kann crashen ("Publishing changes from background threads is not allowed").

7. session.startRunning() auf dem Main-Thread → UI friert ein
Datei: CameraModel.swift

startRunning() ist ein blockierender Aufruf. Da CameraModel @MainActor ist, blockiert das die gesamte UI.

Fix: Auf eine Background-Queue dispatchen.

8. Kamera-Session wird nie gestoppt → Akku-Drain
Datei: CameraModel.swift

In stopCapturing() werden Timer und Motion invalidiert, aber session.stopRunning() wird nie aufgerufen. Die Kamera läuft die gesamte App-Laufzeit.

9. Retain Cycle: CameraModel → photoHandler → CameraModel
Datei: ContentView.swift

Die Closure captured cameraModel strong, und CameraModel speichert die Closure in self.photoHandler → Retain Cycle, Memory Leak.

Fix: [weak cameraModel] verwenden und in stopCapturing() photoHandler = nil setzen.

10. finishGame() wird doppelt aufgerufen bei Sieg
Datei: ContentView.swift

Bei einem Sieg wird finishGame() sofort aufgerufen und nochmal, wenn der User den Button im WinOverlayView drückt. Das Reset passiert doppelt und die UI springt unter dem Overlay.

🟡 Moderate Probleme
11. Bull/Bullseye-Multiplier-Bug in AnalogView
Datei: AnalogView.swift

Wenn man "Triple" drückt und dann "50" (Bull), bekommt man 150 — ein unmöglicher Dart-Score. Bull-Bereiche haben keine Triple/Double-Varianten.

12. Ungenutzter DartTracker in ContentView
Ein separater let dartTracker = DartTracker() wird erzeugt, aber nie verwendet — es wird immer cameraModel.dartTracker benutzt.

13. followMeHighscore wird bei jedem Spielstart zurückgesetzt
GameData() setzt followMeHighscore = 0. Der Highscore geht bei jedem neuen Spiel verloren. Sollte z.B. in @AppStorage persistiert werden.

Prioritäten-Übersicht
Prio	Bug	Auswirkung
🔴 P0	#1 — Erster Dart geht verloren	Falsche Scores jede Runde
🔴 P0	#2 — Bust korrumpiert nächsten Spieler	Falscher Spieler betroffen
🔴 P0	#4 — Observer-Duplikate	Doppelte Score-Abzüge
🔴 P1	#3 — isThrowBusted bleibt true	Tracker-Fehler
🟠 P1	#5, #6 — Background-Thread-Zugriffe	Crashes / Data Races
🟠 P2	#7 — UI-Freeze bei Kamera-Start	Schlechte UX
🟠 P2	#8 — Kamera nie gestoppt	Akku-Drain
🟠 P2	#9 — Retain Cycle	Memory Leak
Soll ich die kritischen Bugs (#1–#4) direkt fixen?
