import { Component, EventEmitter, Output } from '@angular/core';
// Angular app.config injects http Client, should be good for use now
import { FormsModule } from '@angular/forms';


@Component({
  selector: 'app-input-bar',
  imports: [ FormsModule ],
  templateUrl: './input-bar.component.html',
  styleUrl: './input-bar.component.css'
})
export class InputBarComponent {

  prompt = "";

  // event emitter to communicate our prompt to the chat-log.component
  @Output() submitPrompt = new EventEmitter<string>();

  onSubmit() {
    if (!this.prompt.trim()) return;
    this.submitPrompt.emit(this.prompt);
    this.prompt = '';
  }
}
