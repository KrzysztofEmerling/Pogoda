#include <gtk/gtk.h>
#include <cairo.h>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <time.h>
#include <math.h>  

#define MAX_DATA_POINTS 216  // 3 dni * 24 godziny * 3 (co 20 minut)
#define INPUT_FEATURES 5
#define OUTPUT_FEATURES 3
#define PREDICTION_HOURS 24

// selected_columns = ['p (mbar)', 'T (degC)', 'rh (%)', 'Tdew (degC)', 'VPmax (mbar)']
#define P_MEAN 8.92632655
#define P_STD 8.47974514

#define T_MEAN 283.00079882
#define T_STD 8.55663505

#define RH_MEAN 13.13652715 
#define RH_STD 7.51162222

#define TDEW_MEAN 76.48203409
#define TDEW_STD 16.35201742

#define VPMAX_MEAN 9.31004185
#define VPMAX_STD 4.17803453

typedef struct {//-
    time_t timestamp;//-
    double pressure;//-
    double temperature;//-
    double humidity;//-
    double dew_point;//-
    double vpmax;//-
} WeatherData;//-

WeatherData weather_data[MAX_DATA_POINTS];

TF_Graph* graph;
TF_Session* session;

// Funkcja do ładowania modelu
#include <tensorflow/c/c_api.h>
void load_model() {
    graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* options = TF_NewSessionOptions();

    const char* saved_model_file = "../saved_models/transformer_model/model.weights.h5";

    // Wczytaj model z pliku HDF5
    TF_Buffer* run_options = NULL;
    TF_Buffer* meta_graph_def = TF_NewBuffer();
    TF_Graph* graph = TF_NewGraph();

    session = TF_LoadSessionFromSavedModel(options, run_options, saved_model_file, NULL, 0, graph, meta_graph_def, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error loading saved model: %s\n", TF_Message(status));
        exit(1);
    }

    TF_DeleteBuffer(meta_graph_def);
    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(options);
}

// Funkcja do normalizacji danych
void normalize_data(double* input, double* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i*INPUT_FEATURES] = (input[i*INPUT_FEATURES] - P_MEAN) / P_STD;
        output[i*INPUT_FEATURES+1] = (input[i*INPUT_FEATURES+1] - T_MEAN) / T_STD;
        output[i*INPUT_FEATURES+2] = (input[i*INPUT_FEATURES+2] - RH_MEAN) / RH_STD;
        output[i*INPUT_FEATURES+3] = (input[i*INPUT_FEATURES+3] - TDEW_MEAN) / TDEW_STD;
        output[i*INPUT_FEATURES+4] = (input[i*INPUT_FEATURES+4] - VPMAX_MEAN) / VPMAX_STD;
    }
}

// Funkcja do denormalizacji danych
void denormalize_data(double* input, double* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i*OUTPUT_FEATURES] = input[i*OUTPUT_FEATURES] * P_STD + P_MEAN;
        output[i*OUTPUT_FEATURES+1] = input[i*OUTPUT_FEATURES+1] * T_STD + T_MEAN;
        output[i*OUTPUT_FEATURES+2] = input[i*OUTPUT_FEATURES+2] * RH_STD + RH_MEAN;
    }
}

// Funkcja do wykonania predykcji
void predict_weather(double* input_data, double* output_data) {
    TF_Status* status = TF_NewStatus();
    
    // Przygotowanie danych wejściowych
    int64_t dims[] = {1, MAX_DATA_POINTS, INPUT_FEATURES};
    TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, dims, 3, MAX_DATA_POINTS * INPUT_FEATURES * sizeof(float));
    float* tensor_data = (float*)TF_TensorData(input_tensor);
    
    for (int i = 0; i < MAX_DATA_POINTS * INPUT_FEATURES; i++) {
        tensor_data[i] = (float)input_data[i];
    }
    
    // Wykonanie predykcji
    TF_Output input_op = {TF_GraphOperationByName(graph, "serving_default_input_1"), 0};
    TF_Output output_op = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};
    
    TF_Tensor* output_tensor;
    TF_SessionRun(session, NULL,
                  &input_op, &input_tensor, 1,
                  &output_op, &output_tensor, 1,
                  NULL, 0,
                  NULL, status);
    
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error running session: %s\n", TF_Message(status));
        exit(1);
    }
    
    // Kopiowanie wyników
    float* output_data_raw = (float*)TF_TensorData(output_tensor);
    for (int i = 0; i < PREDICTION_HOURS * OUTPUT_FEATURES; i++) {
        output_data[i] = (double)output_data_raw[i];
    }
    
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);
    TF_DeleteStatus(status);
}

// Globalne tablice do przechowywania przewidywań
double predicted_pressure[PREDICTION_HOURS];
double predicted_temperature[PREDICTION_HOURS];
double predicted_humidity[PREDICTION_HOURS];

// Funkcja do aktualizacji danych pogodowych o predykcje
void update_weather_data_with_prediction() {
    double normalized_input[MAX_DATA_POINTS * INPUT_FEATURES];
    double normalized_output[PREDICTION_HOURS * OUTPUT_FEATURES];
    double denormalized_output[PREDICTION_HOURS * OUTPUT_FEATURES];
    
    // Przygotowanie danych wejściowych
    for (int i = 0; i < MAX_DATA_POINTS; i++) {
        normalized_input[i*INPUT_FEATURES] = weather_data[i].pressure;
        normalized_input[i*INPUT_FEATURES+1] = weather_data[i].temperature;
        normalized_input[i*INPUT_FEATURES+2] = weather_data[i].humidity;
        normalized_input[i*INPUT_FEATURES+3] = weather_data[i].dew_point;
        normalized_input[i*INPUT_FEATURES+4] = weather_data[i].vpmax;
    }
    
    normalize_data(normalized_input, normalized_input, MAX_DATA_POINTS);
    
    // Wykonanie predykcji
    predict_weather(normalized_input, normalized_output);
    
    // Denormalizacja wyników
    denormalize_data(normalized_output, denormalized_output, PREDICTION_HOURS);
    
    // Zapisanie przewidywań do globalnych tablic
    for (int i = 0; i < PREDICTION_HOURS; i++) {
        predicted_pressure[i] = denormalized_output[i*OUTPUT_FEATURES];
        predicted_temperature[i] = denormalized_output[i*OUTPUT_FEATURES+1];
        predicted_humidity[i] = denormalized_output[i*OUTPUT_FEATURES+2];
    }
}

int data_count = 0;//-
// Funkcja do pobierania danych pogodowych (do zaimplementowania)//-
void fetch_weather_data() {//-
    time_t current_time = time(NULL);//-
    struct tm *time_info = localtime(&current_time);//-
//-
    // Ustawienie czasu na początek obecnej godziny//-
    time_info->tm_min = 0;//-
    time_info->tm_sec = 0;//-
    time_t start_time = mktime(time_info);//-
    for (int i = 0; i < MAX_DATA_POINTS; i++) {//-
        weather_data[i].timestamp = start_time - (MAX_DATA_POINTS - 1 - i) * 1200; // co 20 minut//-
        weather_data[i].pressure = 1000 + (rand() % 50);//-
        weather_data[i].temperature = 20 + (rand() % 10);//-
        weather_data[i].humidity = 50 + (rand() % 30);//-
        weather_data[i].dew_point = 15 + (rand() % 5);//-
        weather_data[i].vpmax = 30 + (rand() % 10);//-
    }//-
    data_count = MAX_DATA_POINTS;//-
    printf("Pobrano %d punktów danych.\n", data_count);//-
}//-
void draw_chart(cairo_t *cr, int x, int y, int width, int height, double *data, int data_points, const char *title, int chart_index)
{
    printf("Rysowanie wykresu: %s z %d punktami danych\n", title, data_points);

    // Kolory dla wykresów
    const double colors[3][3] = {
        {0.2, 0.6, 0.9},  // Niebieski
        {0.9, 0.2, 0.2},  // Czerwony
        {0.2, 0.9, 0.2}   // Zielony
    };

    // Rysowanie siatki
    cairo_set_source_rgba(cr, 0.7, 0.7, 0.7, 0.3);
    cairo_set_line_width(cr, 0.5);
    for (int i = 0; i <= 4; i++) {
        double y_pos = y + i * height / 4;
        cairo_move_to(cr, x, y_pos);
        cairo_line_to(cr, x + width, y_pos);
    }
    for (int i = 0; i <= 96; i += 24) {
        double x_pos = x + (double)i / 96 * width;
        cairo_move_to(cr, x_pos, y);
        cairo_line_to(cr, x_pos, y + height);
    }
    cairo_stroke(cr);
    // Rysowanie osi
    cairo_set_source_rgb(cr, 0, 0, 0);
    cairo_set_line_width(cr, 1);
    cairo_move_to(cr, x, y + height);
    cairo_line_to(cr, x, y);
    cairo_move_to(cr, x, y + height);
    cairo_line_to(cr, x + width, y + height);
    cairo_stroke(cr);

    // Rysowanie tytułu
    cairo_move_to(cr, x, y - 10);
    cairo_show_text(cr, title);

    // Rysowanie etykiet osi X
    cairo_set_font_size(cr, 8);  // Zmniejszamy rozmiar czcionki dla lepszej czytelności
    for (int i = 0; i <= 96; i += 6) {  // Zmieniamy krok z 24 na 6 dla etykiet co 6 godzin
        int hour = -72 + i;
        char label[10];
        snprintf(label, sizeof(label), "%d", hour);
        double x_pos = x + (double)i / 96 * width;

        // Rysujemy dłuższe linie dla głównych podziałek (co 24 godziny)
        if (i % 24 == 0) {
            cairo_move_to(cr, x_pos, y + height);
            cairo_line_to(cr, x_pos, y + height + 10);
            cairo_set_source_rgb(cr, 0, 0, 0);  // Czarny kolor dla głównych podziałek
        } else {
            cairo_move_to(cr, x_pos, y + height);
            cairo_line_to(cr, x_pos, y + height + 5);
            cairo_set_source_rgb(cr, 0.5, 0.5, 0.5);  // Szary kolor dla pośrednich podziałek
        }
        cairo_stroke(cr);

        // Rysujemy etykiety tylko dla głównych podziałek
        if (i % 24 == 0) {
            cairo_move_to(cr, x_pos - 10, y + height + 20);
            cairo_set_source_rgb(cr, 0, 0, 0);  // Czarny kolor tekstu
            cairo_show_text(cr, label);
        }
    }

    // Rysowanie punktów i linii
    if (data_points > 0) {
        double max_value = data[0], min_value = data[0];
        for (int i = 1; i < data_points; i++) {
            if (data[i] > max_value) max_value = data[i];
            if (data[i] < min_value) min_value = data[i];
        }

        // Zaokrąglanie min i max do "ładnych" wartości
        min_value = floor(min_value);
        max_value = ceil(max_value);
        printf("Min: %f, Max: %f\n", min_value, max_value);

        // Rysowanie etykiet osi Y
        for (int i = 0; i <= 4; i++) {
            double value = min_value + (max_value - min_value) * i / 4;
            char label[20];
            snprintf(label, sizeof(label), "%.1f", value);
            double y_pos = y + height - i * height / 4;
            cairo_move_to(cr, x - 30, y_pos);
            cairo_show_text(cr, label);
        }
        // Rysowanie linii
        cairo_set_source_rgb(cr, colors[chart_index][0], colors[chart_index][1], colors[chart_index][2]);
        cairo_set_line_width(cr, 2);
        cairo_move_to(cr, x, y + height - (data[0] - min_value) / (max_value - min_value) * height);
        for (int i = 1; i < data_points; i++) {
            double x_pos = x + (double)i / (data_points - 1) * width * 72 / 96;
            double y_pos = y + height - (data[i] - min_value) / (max_value - min_value) * height;
            cairo_line_to(cr, x_pos, y_pos);
        }
        cairo_stroke(cr);

        // Rysowanie punktów
        for (int i = 0; i < data_points; i++) {
            double x_pos = x + (double)i / (data_points - 1) * width * 72 / 96;
            double y_pos = y + height - (data[i] - min_value) / (max_value - min_value) * height;
            cairo_arc(cr, x_pos, y_pos, 3, 0, 2 * M_PI);
            cairo_fill(cr);
        }

        //przewidywania
        cairo_set_source_rgb(cr, 1.0, 0.5, 0.0);  // Pomarańczowy kolor dla przewidywań
        cairo_set_line_width(cr, 2);

        double *predicted_data;
        switch (chart_index) {
            case 0: predicted_data = predicted_pressure; break;
            case 1: predicted_data = predicted_temperature; break;
            case 2: predicted_data = predicted_humidity; break;
            default: return;
        }

        for (int i = 0; i < PREDICTION_HOURS; i++) {
            double x_pos = x + (double)(data_points + i) / (data_points + PREDICTION_HOURS - 1) * width;
            double y_pos = y + height - (predicted_data[i] - min_value) / (max_value - min_value) * height;
            
            if (i == 0) {
                cairo_move_to(cr, x_pos, y_pos);
            } else {
                cairo_line_to(cr, x_pos, y_pos);
            }
        }
        cairo_stroke(cr);

        // Rysowanie punktów dla przewidywań
        for (int i = 0; i < PREDICTION_HOURS; i++) {
            double x_pos = x + (double)(data_points + i) / (data_points + PREDICTION_HOURS - 1) * width;
            double y_pos = y + height - (predicted_data[i] - min_value) / (max_value - min_value) * height;
            cairo_arc(cr, x_pos, y_pos, 3, 0, 2 * M_PI);
            cairo_fill(cr);
        }
    } 
    else {
        printf("Brak danych do wyświetlenia dla %s\n", title);
    }
}

static void draw_weather_charts(GtkDrawingArea *area, cairo_t *cr, int width, int height, gpointer user_data) {
    printf("Wywołano funkcję draw_weather_charts\n");
    int chart_width = width - 40;
    int chart_height = (height - 120) / 3;

    double pressure_data[MAX_DATA_POINTS / 3];
    double temperature_data[MAX_DATA_POINTS / 3];
    double humidity_data[MAX_DATA_POINTS / 3];
    int hourly_data_count = 0;

    // Filtrowanie danych do pełnych godzin
    for (int i = 0; i < data_count; i++) {
        struct tm *time_info = localtime(&weather_data[i].timestamp);
        if (time_info->tm_min == 0) {
            pressure_data[hourly_data_count] = weather_data[i].pressure;
            temperature_data[hourly_data_count] = weather_data[i].temperature;
            humidity_data[hourly_data_count] = weather_data[i].humidity;
            hourly_data_count++;
            printf("Znaleziono dane dla pełnej godziny: %s", ctime(&weather_data[i].timestamp));
        }
    }

    printf("Przefiltrowano %d punktów danych do pełnych godzin\n", hourly_data_count);

    draw_chart(cr, 20, 50, chart_width, chart_height, pressure_data, hourly_data_count, "Ciśnienie (mbar)", 0);
    draw_chart(cr, 20, 50 + chart_height + 40, chart_width, chart_height, temperature_data, hourly_data_count, "Temperatura (°C)", 1);
    draw_chart(cr, 20, 50 + 2 * (chart_height + 40), chart_width, chart_height, humidity_data, hourly_data_count, "Wilgotność (%)", 2);
}

static void activate(GtkApplication *app, gpointer user_data) {
    GtkWidget *window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(window), "Aplikacja pogodowa");
    gtk_window_set_default_size(GTK_WINDOW(window), 800, 600);  // Zmieniony rozmiar okna//+

    GtkWidget *drawing_area = gtk_drawing_area_new();
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(drawing_area), draw_weather_charts, NULL, NULL);

    gtk_window_set_child(GTK_WINDOW(window), drawing_area);

    gtk_widget_set_visible(window, TRUE);
}//-
//-
int main(int argc, char *argv[]) 
{
    fetch_weather_data();  // Pobierz dane pogodowe
    load_model();  // Załaduj model
    update_weather_data_with_prediction();  // Wykonaj predykcję
    
    GtkApplication *app;
    int status;

    app = gtk_application_new("org.gtk.pogoda", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);

    // Zwolnij zasoby modelu
    TF_DeleteSession(session, TF_NewStatus());
    TF_DeleteGraph(graph);

    return status;
}