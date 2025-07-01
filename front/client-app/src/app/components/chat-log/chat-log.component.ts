import { Component, signal } from '@angular/core';
import { NgClass } from '@angular/common';
import { PredictionService } from '../../services/prediction.service';
import { ParsedMessage } from '../../services/parse.service';

// only using it in this one window - for bigger apps this should be modular
interface ChatMessage extends ParsedMessage { id: number; }

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
    this.messages.update(msgs => [...msgs, { id: this.currID++, user: "You", role: "user", content: prompt }]);

    // Make service request; get response based on user message
    this.predictionService.getPrediction({ prompt: prompt }).subscribe({

      // valid
      next: (aiMsgs) => {
        this.messages.update(msgs => [...msgs, ...aiMsgs.map(msg => ({ id: this.currID++, ...msg}) ) ]);
      },

      // Display as error in UI somehow
      error: () => {
        this.messages.update(m => [
            ...m,
            { id: this.currID++, user:'bot', role:'ai', content:'<ERROR>' }
        ]);
      }

    });
  

  }

}
