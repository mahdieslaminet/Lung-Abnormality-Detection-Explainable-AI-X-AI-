import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, PredictionResult } from '../services/api.service';
import { trigger, transition, style, animate } from '@angular/animations';
import { MOCK_DATASET_RESPONSE } from '../data/mock-dataset';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css'],
  animations: [
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(20px)' }),
        animate('0.5s ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
      ])
    ])
  ]
})
export class HomeComponent {
  selectedFile: File | null = null;
  prediction: PredictionResult | null = null;
  loading = false;
  error: string | null = null;
  imagePreview: string | null = null;
  
  // Dataset properties
  datasetHandle = 'luisblanche/covidct';
  loadedDatasetHandle = ''; // Store the handle of the currently displayed data
  datasetPreview: any = null;
  datasetLoading = false;
  datasetError: string | null = null;
  
  // Pagination
  currentPage = 1;
  pageSize = 8;
  
  get totalPages(): number {
    if (!this.datasetPreview || !this.datasetPreview.head) return 0;
    return Math.ceil(this.datasetPreview.head.length / this.pageSize);
  }
  
  get paginatedRows(): any[] {
    if (!this.datasetPreview || !this.datasetPreview.head) return [];
    const start = (this.currentPage - 1) * this.pageSize;
    return this.datasetPreview.head.slice(start, start + this.pageSize);
  }

  constructor(private apiService: ApiService, private cdr: ChangeDetectorRef) {}
  
  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.prediction = null;
      this.error = null;
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreview = reader.result as string;
        this.cdr.detectChanges();
      };
      reader.readAsDataURL(file);
    }
  }

  changePage(page: number) {
    if (page >= 1 && page <= this.totalPages) {
      this.currentPage = page;
    }
  }

  async uploadAndPredict() {
    if (!this.selectedFile) return;

    this.loading = true;
    this.error = null;
    this.prediction = null;
    this.cdr.detectChanges();

    this.apiService.predict(this.selectedFile).subscribe({
      next: (result) => {
        console.log('Prediction Result:', result);
        this.prediction = result;
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error(err);
        this.error = 'خطا در پردازش تصویر. لطفا دوباره تلاش کنید.';
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }

  loadDataset() {
    this.datasetLoading = true;
    this.datasetError = null;
    this.datasetPreview = null;
    
    // Capture the handle being loaded
    const handleToLoad = this.datasetHandle;

    this.apiService.loadDataset(handleToLoad).subscribe({
      next: (data) => {
        console.log('Dataset loaded:', data);
        this.datasetPreview = data;
        this.loadedDatasetHandle = handleToLoad; // Update the confirmed loaded handle
        this.datasetLoading = false;
        this.currentPage = 1; 
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error loading dataset:', err);
        this.datasetError = 'خطا در بارگذاری دیتاست. لطفا هندل دیتاست را بررسی کنید.';
        this.datasetLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  loadMockDataset() {
    this.datasetLoading = true;
    this.datasetError = null;
    this.datasetPreview = null;

    setTimeout(() => {
      console.log('Loading Mock Dataset');
      this.datasetPreview = MOCK_DATASET_RESPONSE;
      this.loadedDatasetHandle = 'luisblanche/covidct'; // Hardcode the mock handle
      this.datasetLoading = false;
      this.currentPage = 1;
      this.cdr.detectChanges();
    }, 500);
  }

  analyzeDatasetImage(filePath: string) {
    this.loading = true;
    this.datasetError = null;
    this.prediction = null;
    
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Use loadedDatasetHandle instead of the input value
    this.apiService.predictDatasetImage(this.loadedDatasetHandle, filePath).subscribe({
      next: (result) => {
        console.log('Prediction Result:', result);
        this.prediction = result;
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error(err);
        this.datasetError = 'خطا در پردازش تصویر انتخاب شده.';
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }

  reset() {
    this.selectedFile = null;
    this.prediction = null;
    this.imagePreview = null;
  }

  getPersianLabel(className: string): string {
    const translations: { [key: string]: string } = {
      'Normal': 'سالم (طبیعی)',
      'Pneumonia': 'ذات‌الریه (Pneumonia)',
      'COVID-19': 'کووید-۱۹ (COVID-19)',
      'Lung Opacity': 'کدورت ریه (Lung Opacity)'
    };
    return translations[className] || className;
  }
}
