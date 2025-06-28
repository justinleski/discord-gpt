import { Component, signal } from '@angular/core';
import { NgClass } from '@angular/common';
import { PredictionService } from '../../services/prediction.service';

// only using it in this one window - for bigger apps this should be modular
type ChatMessage = {
  id: number;
  role: 'user' | 'ai';
  content: string;
};

@Component({
  selector: 'app-chat-log',
  imports: [NgClass],
  templateUrl: './chat-log.component.html',
  styleUrl: './chat-log.component.css'
})
export class ChatLogComponent {

  messages = signal<ChatMessage[]>([]);
  private currID = 0;

  constructor(private predictionService: PredictionService) {}

  // from the service, we will subscribe to any results returned from server as Observable
  handlePrompt(prompt: string) {

    if (!prompt.trim()) return;

    // Add user messages; copy all old messages and add a new user message (role)
    this.messages.update(msgs => [...msgs, { id: this.currID++, role: "user", content: prompt }]);

    // Make service request; get response based on user message
    this.predictionService.getPrediction({ prompt: prompt }).subscribe({

      // valid
      next: (res) => {
        this.messages.update(msgs => [...msgs, { id: this.currID++, role: "ai", content: res.output }]);
      },

      // Display as error in UI somehow
      error: (err) => {
        this.messages.update(msgs => [...msgs, { id: this.currID++, role: "ai", content: "<ERROR>" }]);
      }

    });
  

  }

}
