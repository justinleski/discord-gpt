import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PredictionService {

  private predictionUrl = 'http://localhost:5000/api/generate'; //TODO: make in .env

  constructor(private http: HttpClient) { }

  // Angular returns Observables, this can later be resolved using subscribe to resolve dt
  getPrediction(input: any) : Observable<{ output: string }> {

    // Angular will parse the JSON as JS object once subscribed to
    return this.http.post<{ output: string }>(this.predictionUrl, input);

  }
}
