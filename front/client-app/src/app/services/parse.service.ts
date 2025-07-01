import { Injectable } from '@angular/core';

export interface ParsedMessage {
  user: string;
  role: 'ai' | 'user';
  content: string;
}

@Injectable({
  providedIn: 'root'
})
export class ParseService {

  constructor() { }

  // This function will parse a string. Typically based on input, they do not begin with a user tag
  // so we wil keep going until we see our first <U_NAME>. Then, we will read until the [END] tag 
  parseString(completion: string, role: 'ai' | 'user' = 'ai'): ParsedMessage[] {

    // we want to get rid of everything until the first <U_NAME>
    const start = completion.search(/<([^>]+)>:/);

    if (start === -1) return [];
    const usable = completion.slice(start);


    const lines = usable
      .split(/\[END\]/g)
      .map(l => l.trim())
      .filter(Boolean);

    const count = Math.max(1, Math.floor(Math.random() * 3) + 1);
    const slice = lines.slice(0, count);


    const out: ParsedMessage[] = [];
    slice.forEach(line => {
      const m = line.match(/^<([^>]+)>:\s*(.*)$/s);
      if (!m) return;
      const [ , user, text ] = m;
      const prev = out[out.length - 1];
      if (prev && prev.user === user) {
        prev.content += ' ' + text;
      } else {
        out.push({ user, role, content: text });
      }
    });
    return out;
  }
}
