import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { HeaderComponent } from './components/header/header.component';
import { InputBarComponent } from './components/input-bar/input-bar.component';
import { ChatLogComponent } from './components/chat-log/chat-log.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, InputBarComponent, ChatLogComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',

})
export class AppComponent {
  title = 'client-app';
}
