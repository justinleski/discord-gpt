import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map, Observable } from 'rxjs';
import { ParseService, ParsedMessage } from './parse.service';

@Injectable({
  providedIn: 'root'
})
export class PredictionService {

  private predictionUrl = 'http://localhost:5000/api/generate'; //TODO: make in .env

  // Angualr can have only one constructor; both should be injected in the same constructor
  constructor(
    private http: HttpClient,
    private parseService: ParseService
  ) { }


  // Angular returns Observables, this can later be resolved using subscribe to resolve dt
  getPrediction(body: { prompt: string; max_tokens?: number }): Observable<ParsedMessage[]> {
    return this.http
      .post<{ output: string }>(this.predictionUrl, body)
      .pipe(
      map(res => this.parseService.parseString(res.output, "ai")) // NEXT TIME: make sure you 
    );
  } 
}
